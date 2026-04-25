#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
static unsigned int trailing_zeros_u64(uint64_t x)
{
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (unsigned int)idx;
}
#else
static unsigned int trailing_zeros_u64(uint64_t x)
{
    return (unsigned int)__builtin_ctzll(x);
}
#endif

typedef struct
{
    int n;
    int source;
    int words;
    int INF;
    uint64_t last_mask;
    double delta;
    double base_score;

    uint64_t *adj_bits; /* n x words, shadow graph with source removed */
    uint16_t *dist;     /* n x n distances in shadow graph */

    int *source_out_mult;   /* multiplicity of arcs source -> v */
    int *source_total_mult; /* multiplicity of arcs touching source and v in either direction */

    uint16_t *best1;
    uint16_t *best2;
    int *label1;

    double *pow_delta;
    double *add_delta;
    double *delete_delta;
    double *swap_delta; /* n x n */
} Kernel;

/*
 * Private python-igraph 1.0.0 layout. Keep this boxed in here.
 * The wheel used by this project stores igraph_int_t as int64_t and
 * exposes the graph pointer through Graph.__graph_as_capsule().
 */
typedef int64_t IGraphInt;

typedef struct
{
    IGraphInt *stor_begin;
    IGraphInt *stor_end;
    IGraphInt *end;
} IGraphIntVector;

typedef struct
{
    IGraphInt n;
    IGraphInt directed;
    IGraphIntVector from;
    IGraphIntVector to;
    IGraphIntVector oi;
    IGraphIntVector ii;
    IGraphIntVector os;
    IGraphIntVector is;
    void *attr;
} IGraph1Graph;

typedef struct
{
    int *values;
    Py_ssize_t len;
} VertexList;

static int precompute(Kernel *k);

static inline void *xcalloc(size_t count, size_t size)
{
    if (count == 0 || size == 0)
        return NULL;
    return PyMem_Calloc(count, size);
}

static inline int bit_words(int n)
{
    return (n + 63) >> 6;
}

static inline uint64_t *adj_row(const Kernel *k, int v)
{
    return k->adj_bits + (size_t)v * (size_t)k->words;
}

static inline void set_bit(uint64_t *bits, int v)
{
    bits[v >> 6] |= (uint64_t)1 << (v & 63);
}

static inline int bitset_nonempty(const uint64_t *bits, int words)
{
    for (int i = 0; i < words; ++i)
    {
        if (bits[i] != 0)
            return 1;
    }
    return 0;
}

static inline double weight_from_hdist(const Kernel *k, uint16_t hdist)
{
    if ((int)hdist >= k->INF)
        return 0.0;
    return k->pow_delta[(int)hdist + 1];
}

static void free_kernel(Kernel *k)
{
    PyMem_Free(k->adj_bits);
    PyMem_Free(k->dist);
    PyMem_Free(k->source_out_mult);
    PyMem_Free(k->source_total_mult);
    PyMem_Free(k->best1);
    PyMem_Free(k->best2);
    PyMem_Free(k->label1);
    PyMem_Free(k->pow_delta);
    PyMem_Free(k->add_delta);
    PyMem_Free(k->delete_delta);
    PyMem_Free(k->swap_delta);
    memset(k, 0, sizeof(*k));
}

static int alloc_kernel(Kernel *k)
{
    const size_t n = (size_t)k->n;
    const size_t words = (size_t)k->words;

    k->adj_bits = xcalloc(n * words, sizeof(uint64_t));
    k->dist = xcalloc(n * n, sizeof(uint16_t));
    k->source_out_mult = xcalloc(n, sizeof(int));
    k->source_total_mult = xcalloc(n, sizeof(int));
    k->best1 = xcalloc(n, sizeof(uint16_t));
    k->best2 = xcalloc(n, sizeof(uint16_t));
    k->label1 = xcalloc(n, sizeof(int));
    k->pow_delta = xcalloc(n + 2, sizeof(double));
    k->add_delta = xcalloc(n, sizeof(double));
    k->delete_delta = xcalloc(n, sizeof(double));
    k->swap_delta = xcalloc(n * n, sizeof(double));

    if (!k->adj_bits || !k->dist || !k->source_out_mult || !k->source_total_mult ||
        !k->best1 || !k->best2 || !k->label1 || !k->pow_delta ||
        !k->add_delta || !k->delete_delta || !k->swap_delta)
    {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

static void init_kernel(Kernel *k, int n, int source, double delta)
{
    memset(k, 0, sizeof(*k));
    k->n = n;
    k->source = source;
    k->words = bit_words(n);
    k->INF = n + 1;
    k->last_mask = (n & 63) ? (((uint64_t)1 << (n & 63)) - 1U) : UINT64_MAX;
    k->delta = delta;
}

static void add_directed_edge(Kernel *k, int a, int b)
{
    const int source = k->source;

    if (a == source && b != source)
    {
        k->source_out_mult[b] += 1;
        k->source_total_mult[b] += 1;
    }
    else if (b == source && a != source)
    {
        k->source_total_mult[a] += 1;
    }

    if (a == b || a == source || b == source)
    {
        return;
    }

    set_bit(adj_row(k, a), b);
    set_bit(adj_row(k, b), a);
}

static int graph_int_vector_len(const IGraphIntVector *vec, Py_ssize_t *out)
{
    if (vec->stor_begin == NULL || vec->stor_end == NULL || vec->end == NULL ||
        vec->end < vec->stor_begin || vec->end > vec->stor_end)
    {
        PyErr_SetString(PyExc_ValueError, "invalid igraph integer vector layout");
        return -1;
    }

    const ptrdiff_t len = vec->end - vec->stor_begin;
    if (len < 0 || (uintmax_t)len > (uintmax_t)PY_SSIZE_T_MAX)
    {
        PyErr_SetString(PyExc_OverflowError, "igraph integer vector is too large");
        return -1;
    }

    *out = (Py_ssize_t)len;
    return 0;
}

static int get_igraph_graph(PyObject *graph_obj, const IGraph1Graph **out)
{
    PyObject *method = PyObject_GetAttrString(graph_obj, "__graph_as_capsule");
    if (method == NULL)
    {
        return -1;
    }

    PyObject *capsule = PyObject_CallNoArgs(method);
    Py_DECREF(method);
    if (capsule == NULL)
    {
        return -1;
    }

    void *ptr = PyCapsule_GetPointer(capsule, NULL);
    Py_DECREF(capsule);
    if (ptr == NULL)
    {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, "igraph capsule contained a null graph pointer");
        return -1;
    }

    *out = (const IGraph1Graph *)ptr;
    return 0;
}

static int build_kernel_from_graph(Kernel *k, PyObject *graph_obj, int source, double delta)
{
    const IGraph1Graph *graph;
    Py_ssize_t from_len;
    Py_ssize_t to_len;

    if (get_igraph_graph(graph_obj, &graph) < 0)
        return -1;
    if (graph->n <= 0 || graph->n > INT_MAX)
    {
        PyErr_SetString(PyExc_ValueError, "n must be positive");
        return -1;
    }

    const int n = (int)graph->n;
    if (source < 0 || source >= n)
    {
        PyErr_SetString(PyExc_ValueError, "source out of range");
        return -1;
    }

    init_kernel(k, n, source, delta);
    if (alloc_kernel(k) < 0)
        goto fail;

    if (graph_int_vector_len(&graph->from, &from_len) < 0 ||
        graph_int_vector_len(&graph->to, &to_len) < 0)
    {
        goto fail;
    }
    if (from_len != to_len)
    {
        PyErr_SetString(PyExc_ValueError, "igraph edge vectors have inconsistent lengths");
        goto fail;
    }

    for (Py_ssize_t i = 0; i < from_len; ++i)
    {
        const IGraphInt a_l = graph->from.stor_begin[i];
        const IGraphInt b_l = graph->to.stor_begin[i];
        if (a_l < 0 || a_l >= n || b_l < 0 || b_l >= n)
        {
            PyErr_SetString(PyExc_ValueError, "edge endpoint out of range");
            goto fail;
        }
        add_directed_edge(k, (int)a_l, (int)b_l);
    }

    if (precompute(k) < 0)
        goto fail;

    return 0;

fail:
    free_kernel(k);
    return -1;
}

static int compute_all_pairs(Kernel *k)
{
    const int n = k->n;
    const int source = k->source;
    const uint16_t INF = (uint16_t)k->INF;
    const int words = k->words;

    uint64_t *visited = PyMem_Malloc((size_t)words * sizeof(uint64_t));
    uint64_t *frontier = PyMem_Malloc((size_t)words * sizeof(uint64_t));
    uint64_t *next_frontier = PyMem_Malloc((size_t)words * sizeof(uint64_t));
    if (visited == NULL || frontier == NULL || next_frontier == NULL)
    {
        PyMem_Free(visited);
        PyMem_Free(frontier);
        PyMem_Free(next_frontier);
        PyErr_NoMemory();
        return -1;
    }

    for (int start = 0; start < n; ++start)
    {
        uint16_t *row = k->dist + (size_t)start * (size_t)n;
        for (int v = 0; v < n; ++v)
            row[v] = INF;
        if (start == source)
            continue;

        memset(visited, 0, (size_t)words * sizeof(uint64_t));
        memset(frontier, 0, (size_t)words * sizeof(uint64_t));
        set_bit(visited, start);
        set_bit(frontier, start);
        row[start] = 0;

        uint16_t depth = 0;
        while (bitset_nonempty(frontier, words))
        {
            memset(next_frontier, 0, (size_t)words * sizeof(uint64_t));

            for (int w = 0; w < words; ++w)
            {
                uint64_t bits = frontier[w];
                while (bits != 0)
                {
                    const unsigned int b = trailing_zeros_u64(bits);
                    bits &= bits - 1;
                    const int x = (w << 6) + (int)b;
                    uint64_t *nbrs = adj_row(k, x);
                    for (int q = 0; q < words; ++q)
                    {
                        next_frontier[q] |= nbrs[q];
                    }
                }
            }

            for (int q = 0; q < words; ++q)
            {
                next_frontier[q] &= ~visited[q];
            }
            next_frontier[words - 1] &= k->last_mask;
            if (!bitset_nonempty(next_frontier, words))
            {
                break;
            }

            depth = (uint16_t)(depth + 1);
            for (int w = 0; w < words; ++w)
            {
                uint64_t bits = next_frontier[w];
                while (bits != 0)
                {
                    const unsigned int b = trailing_zeros_u64(bits);
                    bits &= bits - 1;
                    const int y = (w << 6) + (int)b;
                    row[y] = depth;
                }
                visited[w] |= next_frontier[w];
                frontier[w] = next_frontier[w];
            }
        }
    }

    PyMem_Free(visited);
    PyMem_Free(frontier);
    PyMem_Free(next_frontier);
    return 0;
}

static int precompute(Kernel *k)
{
    const int n = k->n;
    const int source = k->source;
    const int INF = k->INF;

    k->pow_delta[0] = 1.0;
    for (int p = 1; p <= n + 1; ++p)
    {
        k->pow_delta[p] = k->pow_delta[p - 1] * k->delta;
    }
    if (compute_all_pairs(k) < 0)
        return -1;

    for (int x = 0; x < n; ++x)
    {
        k->best1[x] = (uint16_t)INF;
        k->best2[x] = (uint16_t)INF;
        k->label1[x] = -1;
        if (x == source)
            continue;
        for (int t = 0; t < n; ++t)
        {
            if (t == source || k->source_total_mult[t] == 0)
                continue;
            const uint16_t d = k->dist[(size_t)t * (size_t)n + (size_t)x];
            if (d < k->best1[x])
            {
                k->best2[x] = k->best1[x];
                k->best1[x] = d;
                k->label1[x] = t;
            }
            else if (d == k->best1[x])
            {
                if (d < k->best2[x])
                    k->best2[x] = d;
            }
            else if (d < k->best2[x])
            {
                k->best2[x] = d;
            }
        }
    }

    k->base_score = 0.0;
    for (int x = 0; x < n; ++x)
    {
        if (x == source)
            continue;
        const double base_w = weight_from_hdist(k, k->best1[x]);
        k->base_score += base_w;
        const int t = k->label1[x];
        if (t >= 0 && k->best2[x] > k->best1[x])
        {
            k->delete_delta[t] += weight_from_hdist(k, k->best2[x]) - base_w;
        }
    }

    for (int u = 0; u < n; ++u)
    {
        if (u == source || k->source_total_mult[u] > 0)
        {
            k->add_delta[u] = 0.0;
            continue;
        }
        const uint16_t *row = k->dist + (size_t)u * (size_t)n;
        double total = 0.0;
        for (int x = 0; x < n; ++x)
        {
            if (x == source)
                continue;
            const uint16_t base_h = k->best1[x];
            const uint16_t new_h = row[x] < base_h ? row[x] : base_h;
            total += weight_from_hdist(k, new_h) - weight_from_hdist(k, base_h);
        }
        k->add_delta[u] = total;
    }

    for (int v_del = 0; v_del < n; ++v_del)
    {
        if (v_del == source || k->source_out_mult[v_del] == 0)
            continue;
        const int removes_shadow = (k->source_total_mult[v_del] == 1);
        for (int v_add = 0; v_add < n; ++v_add)
        {
            if (v_add == v_del)
            {
                k->swap_delta[(size_t)v_del * (size_t)n + (size_t)v_add] = 0.0;
                continue;
            }
            const int adds_shadow = (v_add != source && k->source_total_mult[v_add] == 0);
            if (!removes_shadow && !adds_shadow)
            {
                k->swap_delta[(size_t)v_del * (size_t)n + (size_t)v_add] = 0.0;
                continue;
            }
            if (!removes_shadow && adds_shadow)
            {
                k->swap_delta[(size_t)v_del * (size_t)n + (size_t)v_add] = k->add_delta[v_add];
                continue;
            }
            const uint16_t *row_add = adds_shadow ? (k->dist + (size_t)v_add * (size_t)n) : NULL;
            double total = 0.0;
            for (int x = 0; x < n; ++x)
            {
                if (x == source)
                    continue;
                const uint16_t base_h = k->best1[x];
                uint16_t del_h = base_h;
                if (k->label1[x] == v_del && k->best2[x] > base_h)
                {
                    del_h = k->best2[x];
                }
                uint16_t new_h = del_h;
                if (row_add != NULL && row_add[x] < new_h)
                {
                    new_h = row_add[x];
                }
                total += weight_from_hdist(k, new_h) - weight_from_hdist(k, base_h);
            }
            k->swap_delta[(size_t)v_del * (size_t)n + (size_t)v_add] = total;
        }
    }

    return 0;
}

static int validate_vertex(const Kernel *k, long v_l, int *out)
{
    if (v_l < 0 || v_l >= k->n)
    {
        PyErr_SetString(PyExc_ValueError, "vertex out of range");
        return -1;
    }
    *out = (int)v_l;
    return 0;
}

static void free_vertex_list(VertexList *list)
{
    PyMem_Free(list->values);
    memset(list, 0, sizeof(*list));
}

static int parse_vertex_list(const Kernel *k, PyObject *obj, const char *message, VertexList *out)
{
    PyObject *seq;
    PyObject **items;

    memset(out, 0, sizeof(*out));
    seq = PySequence_Fast(obj, message);
    if (seq == NULL)
        return -1;

    out->len = PySequence_Fast_GET_SIZE(seq);
    if (out->len > 0)
    {
        out->values = PyMem_Malloc((size_t)out->len * sizeof(int));
        if (out->values == NULL)
        {
            PyErr_NoMemory();
            Py_DECREF(seq);
            return -1;
        }
    }

    items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < out->len; ++i)
    {
        long v_l = PyLong_AsLong(items[i]);
        if ((v_l == -1 && PyErr_Occurred()) || validate_vertex(k, v_l, &out->values[i]) < 0)
        {
            Py_DECREF(seq);
            free_vertex_list(out);
            return -1;
        }
    }

    Py_DECREF(seq);
    return 0;
}

static PyObject *build_delete_values(const Kernel *k, const VertexList *deletes)
{
    PyObject *values = PyList_New(deletes->len);
    if (values == NULL)
        return NULL;

    for (Py_ssize_t i = 0; i < deletes->len; ++i)
    {
        const int v = deletes->values[i];
        double value = 0.0;
        if (v != k->source)
        {
            if (k->source_out_mult[v] == 0)
            {
                PyErr_Format(PyExc_ValueError, "No outgoing arc %d->%d exists in the base graph", k->source, v);
                Py_DECREF(values);
                return NULL;
            }
            value = (k->source_total_mult[v] == 1) ? k->delete_delta[v] : 0.0;
        }

        PyObject *value_obj = PyFloat_FromDouble(value);
        if (value_obj == NULL)
        {
            Py_DECREF(values);
            return NULL;
        }
        PyList_SET_ITEM(values, i, value_obj);
    }

    return values;
}

static void best_swap_delta(const Kernel *k, int u, const VertexList *deletes, double *best, int *best_del)
{
    *best = deletes->len > 0 ? -INFINITY : 0.0;
    *best_del = -1;

    for (Py_ssize_t i = 0; i < deletes->len; ++i)
    {
        const int v = deletes->values[i];
        if (v == k->source)
            continue;

        const double cand = k->swap_delta[(size_t)v * (size_t)k->n + (size_t)u];
        if (cand > *best)
        {
            *best = cand;
            *best_del = v;
        }
    }

    if (*best_del == -1 && deletes->len > 0)
        *best = 0.0;
}

static PyObject *evaluate_best_response_terms(PyObject *Py_UNUSED(module), PyObject *args)
{
    Kernel k = {0};
    VertexList adds = {0};
    VertexList deletes = {0};
    PyObject *graph_obj = NULL;
    PyObject *add_targets_obj = NULL;
    PyObject *delete_targets_obj = NULL;
    PyObject *delete_values = NULL;
    PyObject *add_only_values = NULL;
    PyObject *best_swap_values = NULL;
    PyObject *best_swap_delete = NULL;
    PyObject *result = NULL;
    int source;
    double delta;

    if (!PyArg_ParseTuple(args, "OidOO", &graph_obj, &source, &delta, &add_targets_obj, &delete_targets_obj))
        return NULL;
    if (!(delta >= 0.0 && delta <= 1.0))
    {
        PyErr_SetString(PyExc_ValueError, "delta must lie in [0, 1]");
        return NULL;
    }

    if (build_kernel_from_graph(&k, graph_obj, source, delta) < 0)
        goto done;
    if (parse_vertex_list(&k, add_targets_obj, "add_targets must be a sequence of vertices", &adds) < 0)
        goto done;
    if (parse_vertex_list(&k, delete_targets_obj, "delete_targets must be a sequence of vertices", &deletes) < 0)
        goto done;

    delete_values = build_delete_values(&k, &deletes);
    add_only_values = PyList_New(adds.len);
    best_swap_values = PyList_New(adds.len);
    best_swap_delete = PyList_New(adds.len);
    if (delete_values == NULL || add_only_values == NULL || best_swap_values == NULL || best_swap_delete == NULL)
        goto done;

    for (Py_ssize_t i = 0; i < adds.len; ++i)
    {
        const int u = adds.values[i];
        const double add_only = (u == k.source) ? 0.0 : k.add_delta[u];
        double best_swap;
        int best_del;
        best_swap_delta(&k, u, &deletes, &best_swap, &best_del);

        PyObject *add_only_obj = PyFloat_FromDouble(add_only);
        PyObject *best_swap_obj = PyFloat_FromDouble(best_swap);
        PyObject *best_del_obj = PyLong_FromLong(best_del);
        if (add_only_obj == NULL || best_swap_obj == NULL || best_del_obj == NULL)
        {
            Py_XDECREF(add_only_obj);
            Py_XDECREF(best_swap_obj);
            Py_XDECREF(best_del_obj);
            goto done;
        }
        PyList_SET_ITEM(add_only_values, i, add_only_obj);
        PyList_SET_ITEM(best_swap_values, i, best_swap_obj);
        PyList_SET_ITEM(best_swap_delete, i, best_del_obj);
    }

    result = Py_BuildValue("(dOOOO)", k.base_score, delete_values, add_only_values, best_swap_values, best_swap_delete);

done:
    Py_XDECREF(delete_values);
    Py_XDECREF(add_only_values);
    Py_XDECREF(best_swap_values);
    Py_XDECREF(best_swap_delete);
    free_vertex_list(&adds);
    free_vertex_list(&deletes);
    free_kernel(&k);
    return result;
}

static PyMethodDef methods[] = {
    {"evaluate_best_response_terms", (PyCFunction)evaluate_best_response_terms, METH_VARARGS,
     PyDoc_STR("evaluate_best_response_terms(graph, source, delta, add_targets, delete_targets) -> (base_score, delete_values, add_only_values, best_swap_values, best_swap_delete)")},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_best_response_kernel",
    .m_doc = "Bitset C kernel for source-edge decay-centrality changes",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__best_response_kernel(void)
{
    return PyModule_Create(&moduledef);
}
