#include <python3.4/Python.h>
#include <stdlib.h>

/***********************************************/
/* 1) code to route events to Python object    */
/* note that we could run strings here instead */
/***********************************************/


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

/*****************************************************/
/* 2) python extension module to register handlers   */
/* python imports this module to set handler objects */
/*****************************************************/

void setHandler(PyObject *arg)
{
    /* save Python callable object */
    Handler = arg;
    //Py_XDECREF(Handler);                 /* called before? */
    //PyArg_Parse(arg, "(O)", &Handler);  /* one argument */
    //Py_XINCREF(Handler);                 /* add a reference */
}

void triggerEvent(int arg)
{
    /* let Python simulate event caught by C */
    Route_Event(arg);
}
};

