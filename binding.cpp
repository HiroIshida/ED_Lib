#include <array>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "EDCircles.h"

using namespace cv;

struct _Ellipse {
  _Ellipse(const mEllipse &e) : 
    center({e.center.x, e.center.y}),
    axes({e.axes.width, e.axes.height}),
    theta(e.theta) {}
  std::array<double, 2> center;
  std::array<int, 2> axes;
  double theta;
};

class _EDCircles : public EDCircles {
  public:
  _EDCircles(const std::string &filename) : EDCircles(imread(filename, 0)) {}
  std::vector<_Ellipse> getEllipses() {
    std::vector<_Ellipse> ellipses;
    for (const auto &e : EDCircles::getEllipses()) {
      ellipses.push_back(_Ellipse(e));
    }
    return ellipses;
  }
};

namespace py = pybind11;

PYBIND11_MODULE(_edlib, m) {
  py::class_<_Ellipse>(m, "Ellipse")
    .def_readonly("center", &_Ellipse::center)
    .def_readonly("axes", &_Ellipse::axes)
    .def_readonly("theta", &_Ellipse::theta);
  py::class_<_EDCircles>(m, "EDCircles")
    .def(py::init<const std::string &>())
    .def("get_ellipses", &_EDCircles::getEllipses);
}
