// SPDX-License-Identifier: Apache-2.0
// Minimal Python binding to the Boost-licensed Clipper 6.4.2 library.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "clipper.hpp"
#include <cmath>
#include <exception>
using namespace ClipperLib;
static PyObject* offset(PyObject*, PyObject* args) {
    PyObject* input; double distance;
    if (!PyArg_ParseTuple(args, "Od", &input, &distance)) return nullptr;
    // Stay well inside Clipper's signed 64-bit coordinate range.
    const double limit = 1152921504606846976.0;  // 2^60
    if (!std::isfinite(distance) || std::fabs(distance) > limit) {
        PyErr_SetString(PyExc_ValueError, "offset distance is outside the safe range");
        return nullptr;
    }
    PyObject* points = PySequence_Fast(input, "points must be a sequence");
    if (!points) return nullptr;
    Path path;
    try {
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(points);
        path.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i=0;i<n;++i) {
            PyObject* point=PySequence_Fast(PySequence_Fast_GET_ITEM(points,i),"point must be a sequence");
            if (!point) {Py_DECREF(points);return nullptr;}
            if (PySequence_Fast_GET_SIZE(point)<2) {Py_DECREF(point);Py_DECREF(points);PyErr_SetString(PyExc_IndexError,"point needs two coordinates");return nullptr;}
            PyObject* x=PyNumber_Long(PySequence_Fast_GET_ITEM(point,0));
            if (!x) {Py_DECREF(point);Py_DECREF(points);return nullptr;}
            PyObject* y=PyNumber_Long(PySequence_Fast_GET_ITEM(point,1));
            Py_DECREF(point);
            if (!y) {Py_DECREF(x);Py_DECREF(points);return nullptr;}
            long long xx=PyLong_AsLongLong(x);long long yy=PyLong_AsLongLong(y);
            Py_DECREF(x);Py_DECREF(y);
            if (PyErr_Occurred()) {Py_DECREF(points);return nullptr;}
            if (xx < -static_cast<long long>(limit) || xx > static_cast<long long>(limit) ||
                yy < -static_cast<long long>(limit) || yy > static_cast<long long>(limit)) {
                Py_DECREF(points);PyErr_SetString(PyExc_ValueError,"coordinate is outside the safe range");return nullptr;
            }
            path.push_back(IntPoint(xx,yy));
        }
        Py_DECREF(points);points=nullptr;
        ClipperOffset clipper;clipper.AddPath(path,jtRound,etClosedPolygon);
        Paths solution;clipper.Execute(solution,distance);
        PyObject* result=PyList_New(static_cast<Py_ssize_t>(solution.size()));
        if (!result) return nullptr;
        for(size_t i=0;i<solution.size();++i) {
            PyObject* poly=PyList_New(static_cast<Py_ssize_t>(solution[i].size()));
            if (!poly) {Py_DECREF(result);return nullptr;}
            PyList_SET_ITEM(result,static_cast<Py_ssize_t>(i),poly);
            for(size_t j=0;j<solution[i].size();++j) {
                PyObject* point=Py_BuildValue("[LL]",solution[i][j].X,solution[i][j].Y);
                if(!point) {Py_DECREF(result);return nullptr;}
                PyList_SET_ITEM(poly,static_cast<Py_ssize_t>(j),point);
            }
        }
        return result;
    } catch(const std::exception& exc) {
        Py_XDECREF(points);PyErr_SetString(PyExc_RuntimeError,exc.what());return nullptr;
    } catch(...) {
        Py_XDECREF(points);PyErr_SetString(PyExc_RuntimeError,"Clipper offset failed");return nullptr;
    }
}
static PyMethodDef methods[]={{"offset",offset,METH_VARARGS,"Expand a polygon using Clipper."},{nullptr,nullptr,0,nullptr}};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_clipper_native",nullptr,-1,methods};
PyMODINIT_FUNC PyInit__clipper_native(){return PyModule_Create(&module);}
