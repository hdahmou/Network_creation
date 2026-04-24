#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
static unsigned int trailing_zeros_u64(uint64_t x) {
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (unsigned int)idx;
}
#else
static unsigned int trailing_zeros_u64(uint64_t x) {
    return (unsigned int)__builtin_ctzll(x);
}
#endif

typedef struct {
    int n;
    int source;
    int words;
    int INF;
    uint64_t last_mask;
    double delta;
    double base_score;

    uint64_t *adj_bits;      /* n x words, shadow graph with source removed */
    uint16_t *dist;          /* n x n distances in shadow graph */

    int *source_out_mult;    /* multiplicity of arcs source -> v */
    int *source_total_mult;  /* multiplicity of arcs touching source and v in either direction */

    uint16_t *best1;
    uint16_t *best2;
    int *label1;

    double *pow_delta;
    double *add_delta;
    double *delete_delta;
    double *swap_delta;      /* n x n */
} Kernel;

static inline void *xcalloc(size_t count, size_t size) {
    if (count == 0 || size == 0) return NULL;
    return PyMem_Calloc(count, size);
}

static inline int bit_words(int n) {
    return (n + 63) >> 6;
}

static inline uint64_t *adj_row(const Kernel *k, int v) {
    return k->adj_bits + (size_t)v * (size_t)k->words;
}

static inline void set_bit(uint64_t *bits, int v) {
    bits[v >> 6] |= (uint64_t)1 << (v & 63);
}

static inline int bitset_nonempty(const uint64_t *bits, int words) {
    for (int i = 0; i < words; ++i) {
        if (bits[i] != 0) return 1;
    }
    return 0;
}

static inline double weight_from_hdist(const Kernel *k, uint16_t hdist) {
    if ((int)hdist >= k->INF) return 0.0;
    return k->pow_delta[(int)hdist + 1];
}

static void free_kernel(Kernel *k) {
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

static int alloc_kernel(Kernel *k) {
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
        !k->add_delta || !k->delete_delta || !k->swap_delta) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

static int parse_edges(Kernel *k, PyObject *edges_obj) {
    PyObject *seq = PySequence_Fast(edges_obj, "directed_edges must be a sequence of pairs");
    if (seq == NULL) return -1;

    const Py_ssize_t m = PySequence_Fast_GET_SIZE(seq);
    PyObject **items = PySequence_Fast_ITEMS(seq);
    const int n = k->n;
    const int source = k->source;

    for (Py_ssize_t i = 0; i < m; ++i) {
        PyObject *pair = PySequence_Fast(items[i], "each edge must be a pair");
        if (pair == NULL) {
            Py_DECREF(seq);
            return -1;
        }
        if (PySequence_Fast_GET_SIZE(pair) != 2) {
            Py_DECREF(pair);
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "each edge must have exactly two endpoints");
            return -1;
        }

        PyObject **ep = PySequence_Fast_ITEMS(pair);
        long a_l = PyLong_AsLong(ep[0]);
        if (a_l == -1 && PyErr_Occurred()) {
            Py_DECREF(pair);
            Py_DECREF(seq);
            return -1;
        }
        long b_l = PyLong_AsLong(ep[1]);
        if (b_l == -1 && PyErr_Occurred()) {
            Py_DECREF(pair);
            Py_DECREF(seq);
            return -1;
        }
        Py_DECREF(pair);

        if (a_l < 0 || a_l >= n || b_l < 0 || b_l >= n) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "edge endpoint out of range");
            return -1;
        }

        const int a = (int)a_l;
        const int b = (int)b_l;

        if (a == source && b != source) {
            k->source_out_mult[b] += 1;
            k->source_total_mult[b] += 1;
        } else if (b == source && a != source) {
            k->source_total_mult[a] += 1;
        }

        if (a == b || a == source || b == source) {
            continue;
        }

        set_bit(adj_row(k, a), b);
        set_bit(adj_row(k, b), a);
    }

    Py_DECREF(seq);
    return 0;
}

static int compute_all_pairs(Kernel *k) {
    const int n = k->n;
    const int source = k->source;
    const uint16_t INF = (uint16_t)k->INF;
    const int words = k->words;

    uint64_t *visited = PyMem_Malloc((size_t)words * sizeof(uint64_t));
    uint64_t *frontier = PyMem_Malloc((size_t)words * sizeof(uint64_t));
    uint64_t *next_frontier = PyMem_Malloc((size_t)words * sizeof(uint64_t));
    if (visited == NULL || frontier == NULL || next_frontier == NULL) {
        PyMem_Free(visited);
        PyMem_Free(frontier);
        PyMem_Free(next_frontier);
        PyErr_NoMemory();
        return -1;
    }

    for (int start = 0; start < n; ++start) {
        uint16_t *row = k->dist + (size_t)start * (size_t)n;
        for (int v = 0; v < n; ++v) row[v] = INF;
        if (start == source) continue;

        memset(visited, 0, (size_t)words * sizeof(uint64_t));
        memset(frontier, 0, (size_t)words * sizeof(uint64_t));
        set_bit(visited, start);
        set_bit(frontier, start);
        row[start] = 0;

        uint16_t depth = 0;
        while (bitset_nonempty(frontier, words)) {
            memset(next_frontier, 0, (size_t)words * sizeof(uint64_t));

            for (int w = 0; w < words; ++w) {
                uint64_t bits = frontier[w];
                while (bits != 0) {
                    const unsigned int b = trailing_zeros_u64(bits);
                    bits &= bits - 1;
                    const int x = (w << 6) + (int)b;
                    uint64_t *nbrs = adj_row(k, x);
                    for (int q = 0; q < words; ++q) {
                        next_frontier[q] |= nbrs[q];
                    }
                }
            }

            for (int q = 0; q < words; ++q) {
                next_frontier[q] &= ~visited[q];
            }
            next_frontier[words - 1] &= k->last_mask;
            if (!bitset_nonempty(next_frontier, words)) {
                break;
            }

            depth = (uint16_t)(depth + 1);
            for (int w = 0; w < words; ++w) {
                uint64_t bits = next_frontier[w];
                while (bits != 0) {
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

static int precompute(Kernel *k) {
    const int n = k->n;
    const int source = k->source;
    const int INF = k->INF;

    k->pow_delta[0] = 1.0;
    for (int p = 1; p <= n + 1; ++p) {
        k->pow_delta[p] = k->pow_delta[p - 1] * k->delta;
    }
    if (compute_all_pairs(k) < 0) return -1;

    for (int x = 0; x < n; ++x) {
        k->best1[x] = (uint16_t)INF;
        k->best2[x] = (uint16_t)INF;
        k->label1[x] = -1;
        if (x == source) continue;
        for (int t = 0; t < n; ++t) {
            if (t == source || k->source_total_mult[t] == 0) continue;
            const uint16_t d = k->dist[(size_t)t * (size_t)n + (size_t)x];
            if (d < k->best1[x]) {
                k->best2[x] = k->best1[x];
                k->best1[x] = d;
                k->label1[x] = t;
            } else if (d == k->best1[x]) {
                if (d < k->best2[x]) k->best2[x] = d;
            } else if (d < k->best2[x]) {
                k->best2[x] = d;
            }
        }
    }

    k->base_score = 0.0;
    for (int x = 0; x < n; ++x) {
        if (x == source) continue;
        const double base_w = weight_from_hdist(k, k->best1[x]);
        k->base_score += base_w;
        const int t = k->label1[x];
        if (t >= 0 && k->best2[x] > k->best1[x]) {
            k->delete_delta[t] += weight_from_hdist(k, k->best2[x]) - base_w;
        }
    }

    for (int u = 0; u < n; ++u) {
        if (u == source || k->source_total_mult[u] > 0) {
            k->add_delta[u] = 0.0;
            continue;
        }
        const uint16_t *row = k->dist + (size_t)u * (size_t)n;
        double total = 0.0;
        for (int x = 0; x < n; ++x) {
            if (x == source) continue;
            const uint16_t base_h = k->best1[x];
            const uint16_t new_h = row[x] < base_h ? row[x] : base_h;
            total += weight_from_hdist(k, new_h) - weight_from_hdist(k, base_h);
        }
        k->add_delta[u] = total;
    }

    for (int v_del = 0; v_del < n; ++v_del) {
        if (v_del == source || k->source_out_mult[v_del] == 0) continue;
        const int removes_shadow = (k->source_total_mult[v_del] == 1);
        for (int v_add = 0; v_add < n; ++v_add) {
            if (v_add == v_del) {
                k->swap_delta[(size_t)v_del * (size_t)n + (size_t)v_add] = 0.0;
                continue;
            }
            const int adds_shadow = (v_add != source && k->source_total_mult[v_add] == 0);
            if (!removes_shadow && !adds_shadow) {
                k->swap_delta[(size_t)v_del * (size_t)n + (size_t)v_add] = 0.0;
                continue;
            }
            if (!removes_shadow && adds_shadow) {
                k->swap_delta[(size_t)v_del * (size_t)n + (size_t)v_add] = k->add_delta[v_add];
                continue;
            }
            const uint16_t *row_add = adds_shadow ? (k->dist + (size_t)v_add * (size_t)n) : NULL;
            double total = 0.0;
            for (int x = 0; x < n; ++x) {
                if (x == source) continue;
                const uint16_t base_h = k->best1[x];
                uint16_t del_h = base_h;
                if (k->label1[x] == v_del && k->best2[x] > base_h) {
                    del_h = k->best2[x];
                }
                uint16_t new_h = del_h;
                if (row_add != NULL && row_add[x] < new_h) {
                    new_h = row_add[x];
                }
                total += weight_from_hdist(k, new_h) - weight_from_hdist(k, base_h);
            }
            k->swap_delta[(size_t)v_del * (size_t)n + (size_t)v_add] = total;
        }
    }

    return 0;
}

static int validate_vertex(const Kernel *k, long v_l, int *out) {
    if (v_l < 0 || v_l >= k->n) {
        PyErr_SetString(PyExc_ValueError, "vertex out of range");
        return -1;
    }
    *out = (int)v_l;
    return 0;
}

static PyObject *evaluate_from_edges(PyObject *Py_UNUSED(module), PyObject *args) {
    int n;
    int source;
    double delta;
    PyObject *edges_obj;
    PyObject *add_targets_obj;
    PyObject *delete_targets_obj;

    if (!PyArg_ParseTuple(args, "iOidOO", &n, &edges_obj, &source, &delta, &add_targets_obj, &delete_targets_obj)) {
        return NULL;
    }
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n must be positive");
        return NULL;
    }
    if (source < 0 || source >= n) {
        PyErr_SetString(PyExc_ValueError, "source out of range");
        return NULL;
    }
    if (!(delta >= 0.0 && delta <= 1.0)) {
        PyErr_SetString(PyExc_ValueError, "delta must lie in [0, 1]");
        return NULL;
    }

    Kernel k;
    memset(&k, 0, sizeof(k));
    k.n = n;
    k.source = source;
    k.words = bit_words(n);
    k.INF = n + 1;
    k.last_mask = (n & 63) ? (((uint64_t)1 << (n & 63)) - 1U) : UINT64_MAX;
    k.delta = delta;

    if (alloc_kernel(&k) < 0) {
        free_kernel(&k);
        return NULL;
    }
    if (parse_edges(&k, edges_obj) < 0) {
        free_kernel(&k);
        return NULL;
    }
    if (precompute(&k) < 0) {
        free_kernel(&k);
        return NULL;
    }

    PyObject *add_seq = PySequence_Fast(add_targets_obj, "add_targets must be a sequence of vertices");
    if (add_seq == NULL) {
        free_kernel(&k);
        return NULL;
    }
    PyObject *del_seq = PySequence_Fast(delete_targets_obj, "delete_targets must be a sequence of vertices");
    if (del_seq == NULL) {
        Py_DECREF(add_seq);
        free_kernel(&k);
        return NULL;
    }

    const Py_ssize_t n_add = PySequence_Fast_GET_SIZE(add_seq);
    const Py_ssize_t n_del = PySequence_Fast_GET_SIZE(del_seq);
    PyObject **add_items = PySequence_Fast_ITEMS(add_seq);
    PyObject **del_items = PySequence_Fast_ITEMS(del_seq);

    PyObject *delete_values = PyList_New(n_del);
    PyObject *add_values = PyList_New(n_add);
    PyObject *add_best_delete = PyList_New(n_add);
    if (delete_values == NULL || add_values == NULL || add_best_delete == NULL) {
        Py_XDECREF(delete_values);
        Py_XDECREF(add_values);
        Py_XDECREF(add_best_delete);
        Py_DECREF(add_seq);
        Py_DECREF(del_seq);
        free_kernel(&k);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n_del; ++i) {
        long v_l = PyLong_AsLong(del_items[i]);
        int v;
        if ((v_l == -1 && PyErr_Occurred()) || validate_vertex(&k, v_l, &v) < 0) {
            Py_DECREF(delete_values);
            Py_DECREF(add_values);
            Py_DECREF(add_best_delete);
            Py_DECREF(add_seq);
            Py_DECREF(del_seq);
            free_kernel(&k);
            return NULL;
        }
        double value = 0.0;
        if (v != source) {
            if (k.source_out_mult[v] == 0) {
                PyErr_Format(PyExc_ValueError, "No outgoing arc %d->%d exists in the base graph", source, v);
                Py_DECREF(delete_values);
                Py_DECREF(add_values);
                Py_DECREF(add_best_delete);
                Py_DECREF(add_seq);
                Py_DECREF(del_seq);
                free_kernel(&k);
                return NULL;
            }
            if (k.source_total_mult[v] == 1) {
                value = k.delete_delta[v];
            }
        }
        PyList_SET_ITEM(delete_values, i, PyFloat_FromDouble(value));
    }

    for (Py_ssize_t i = 0; i < n_add; ++i) {
        long u_l = PyLong_AsLong(add_items[i]);
        int u;
        if ((u_l == -1 && PyErr_Occurred()) || validate_vertex(&k, u_l, &u) < 0) {
            Py_DECREF(delete_values);
            Py_DECREF(add_values);
            Py_DECREF(add_best_delete);
            Py_DECREF(add_seq);
            Py_DECREF(del_seq);
            free_kernel(&k);
            return NULL;
        }
        double best = (u == source) ? 0.0 : k.add_delta[u];
        int best_del = -1;
        for (Py_ssize_t j = 0; j < n_del; ++j) {
            long v_l = PyLong_AsLong(del_items[j]);
            int v = (int)v_l;
            if (v == source) continue;
            if (k.source_out_mult[v] == 0) {
                PyErr_Format(PyExc_ValueError, "No outgoing arc %d->%d exists in the base graph", source, v);
                Py_DECREF(delete_values);
                Py_DECREF(add_values);
                Py_DECREF(add_best_delete);
                Py_DECREF(add_seq);
                Py_DECREF(del_seq);
                free_kernel(&k);
                return NULL;
            }
            const double cand = k.swap_delta[(size_t)v * (size_t)n + (size_t)u];
            if (cand > best) {
                best = cand;
                best_del = v;
            }
        }
        PyList_SET_ITEM(add_values, i, PyFloat_FromDouble(best));
        PyList_SET_ITEM(add_best_delete, i, PyLong_FromLong(best_del));
    }

    const double base_score = k.base_score;
    Py_DECREF(add_seq);
    Py_DECREF(del_seq);
    free_kernel(&k);
    return Py_BuildValue("dOOO", base_score, delete_values, add_values, add_best_delete);
}


static PyObject *evaluate_best_response_terms_from_edges(PyObject *Py_UNUSED(module), PyObject *args) {
    int n;
    int source;
    double delta;
    PyObject *edges_obj;
    PyObject *add_targets_obj;
    PyObject *delete_targets_obj;

    if (!PyArg_ParseTuple(args, "iOidOO", &n, &edges_obj, &source, &delta, &add_targets_obj, &delete_targets_obj)) {
        return NULL;
    }
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n must be positive");
        return NULL;
    }
    if (source < 0 || source >= n) {
        PyErr_SetString(PyExc_ValueError, "source out of range");
        return NULL;
    }
    if (!(delta >= 0.0 && delta <= 1.0)) {
        PyErr_SetString(PyExc_ValueError, "delta must lie in [0, 1]");
        return NULL;
    }

    Kernel k;
    memset(&k, 0, sizeof(k));
    k.n = n;
    k.source = source;
    k.words = bit_words(n);
    k.INF = n + 1;
    k.last_mask = (n & 63) ? (((uint64_t)1 << (n & 63)) - 1U) : UINT64_MAX;
    k.delta = delta;

    if (alloc_kernel(&k) < 0) {
        free_kernel(&k);
        return NULL;
    }
    if (parse_edges(&k, edges_obj) < 0) {
        free_kernel(&k);
        return NULL;
    }
    if (precompute(&k) < 0) {
        free_kernel(&k);
        return NULL;
    }

    PyObject *add_seq = PySequence_Fast(add_targets_obj, "add_targets must be a sequence of vertices");
    if (add_seq == NULL) {
        free_kernel(&k);
        return NULL;
    }
    PyObject *del_seq = PySequence_Fast(delete_targets_obj, "delete_targets must be a sequence of vertices");
    if (del_seq == NULL) {
        Py_DECREF(add_seq);
        free_kernel(&k);
        return NULL;
    }

    const Py_ssize_t n_add = PySequence_Fast_GET_SIZE(add_seq);
    const Py_ssize_t n_del = PySequence_Fast_GET_SIZE(del_seq);
    PyObject **add_items = PySequence_Fast_ITEMS(add_seq);
    PyObject **del_items = PySequence_Fast_ITEMS(del_seq);

    PyObject *delete_values = PyList_New(n_del);
    PyObject *add_only_values = PyList_New(n_add);
    PyObject *best_swap_values = PyList_New(n_add);
    PyObject *best_swap_delete = PyList_New(n_add);
    if (delete_values == NULL || add_only_values == NULL || best_swap_values == NULL || best_swap_delete == NULL) {
        Py_XDECREF(delete_values);
        Py_XDECREF(add_only_values);
        Py_XDECREF(best_swap_values);
        Py_XDECREF(best_swap_delete);
        Py_DECREF(add_seq);
        Py_DECREF(del_seq);
        free_kernel(&k);
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n_del; ++i) {
        long v_l = PyLong_AsLong(del_items[i]);
        int v;
        if ((v_l == -1 && PyErr_Occurred()) || validate_vertex(&k, v_l, &v) < 0) {
            Py_DECREF(delete_values);
            Py_DECREF(add_only_values);
            Py_DECREF(best_swap_values);
            Py_DECREF(best_swap_delete);
            Py_DECREF(add_seq);
            Py_DECREF(del_seq);
            free_kernel(&k);
            return NULL;
        }
        double value = 0.0;
        if (v != source) {
            if (k.source_out_mult[v] == 0) {
                PyErr_Format(PyExc_ValueError, "No outgoing arc %d->%d exists in the base graph", source, v);
                Py_DECREF(delete_values);
                Py_DECREF(add_only_values);
                Py_DECREF(best_swap_values);
                Py_DECREF(best_swap_delete);
                Py_DECREF(add_seq);
                Py_DECREF(del_seq);
                free_kernel(&k);
                return NULL;
            }
            if (k.source_total_mult[v] == 1) {
                value = k.delete_delta[v];
            }
        }
        PyList_SET_ITEM(delete_values, i, PyFloat_FromDouble(value));
    }

    for (Py_ssize_t i = 0; i < n_add; ++i) {
        long u_l = PyLong_AsLong(add_items[i]);
        int u;
        if ((u_l == -1 && PyErr_Occurred()) || validate_vertex(&k, u_l, &u) < 0) {
            Py_DECREF(delete_values);
            Py_DECREF(add_only_values);
            Py_DECREF(best_swap_values);
            Py_DECREF(best_swap_delete);
            Py_DECREF(add_seq);
            Py_DECREF(del_seq);
            free_kernel(&k);
            return NULL;
        }

        const double add_only = (u == source) ? 0.0 : k.add_delta[u];
        double best_swap = 0.0;
        int best_del = -1;
        if (n_del > 0) {
            best_swap = -INFINITY;
            for (Py_ssize_t j = 0; j < n_del; ++j) {
                long v_l = PyLong_AsLong(del_items[j]);
                int v = (int)v_l;
                if (v == source) continue;
                if (k.source_out_mult[v] == 0) {
                    PyErr_Format(PyExc_ValueError, "No outgoing arc %d->%d exists in the base graph", source, v);
                    Py_DECREF(delete_values);
                    Py_DECREF(add_only_values);
                    Py_DECREF(best_swap_values);
                    Py_DECREF(best_swap_delete);
                    Py_DECREF(add_seq);
                    Py_DECREF(del_seq);
                    free_kernel(&k);
                    return NULL;
                }
                const double cand = k.swap_delta[(size_t)v * (size_t)n + (size_t)u];
                if (cand > best_swap) {
                    best_swap = cand;
                    best_del = v;
                }
            }
            if (best_del == -1) {
                best_swap = 0.0;
            }
        }

        PyList_SET_ITEM(add_only_values, i, PyFloat_FromDouble(add_only));
        PyList_SET_ITEM(best_swap_values, i, PyFloat_FromDouble(best_swap));
        PyList_SET_ITEM(best_swap_delete, i, PyLong_FromLong(best_del));
    }

    const double base_score = k.base_score;
    Py_DECREF(add_seq);
    Py_DECREF(del_seq);
    free_kernel(&k);
    return Py_BuildValue("dOOOO", base_score, delete_values, add_only_values, best_swap_values, best_swap_delete);
}

static PyMethodDef methods[] = {
    {"evaluate_from_edges", (PyCFunction)evaluate_from_edges, METH_VARARGS,
     PyDoc_STR("evaluate_from_edges(n, directed_edges, source, delta, add_targets, delete_targets) -> (base_score, delete_values, add_values, add_best_delete)")},
    {"evaluate_best_response_terms_from_edges", (PyCFunction)evaluate_best_response_terms_from_edges, METH_VARARGS,
     PyDoc_STR("evaluate_best_response_terms_from_edges(n, directed_edges, source, delta, add_targets, delete_targets) -> (base_score, delete_values, add_only_values, best_swap_values, best_swap_delete)")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_best_response_kernel",
    .m_doc = "Bitset C kernel for source-edge decay-centrality changes",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__best_response_kernel(void) {
    return PyModule_Create(&moduledef);
}
