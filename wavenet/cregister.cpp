#include <python3.4/Python.h>
#include <stdlib.h>


class Caller{
private:

PyObject*  Route_Event(int count)
{
    PyObject *args, *pres;
    /* call Python handler */
    args = Py_BuildValue("(i)", count);   /* make arg-list */
    pres = PyEval_CallObject(Handler, args);      /* apply: run a call */
    return pres;
}
public:
  PyObject *Handler;     /* keep Python object in C */
  Caller()
  {
    Handler = NULL;     /* keep Python object in C */    
  }


void setHandler(PyObject *arg)
{
    Handler = arg;
}

void triggerEvent(int arg)
{
    Route_Event(arg);
}
};

