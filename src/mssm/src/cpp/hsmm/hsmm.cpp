#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_llk(py::module_ &);
void init_decode(py::module_ &);
void init_resid(py::module_ &);

PYBIND11_MODULE(hsmm, m) {
    init_llk(m);
    init_decode(m);
    init_resid(m);
}