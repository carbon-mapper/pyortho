%module sunpos
%{
   #define SWIG_FILE_WITH_INIT
   #include "SunPos.h"
%}

%include "numpy.i"

%init
%{
   import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {(int len1, double* vec1),
     (int len2, double* vec2),(int len3, double* vec3),(int len4, double* vec4)}

%include "SunPos.h"

%inline
%{
    void sunpos(int yy, int mm, int dd, int h, int m, int s,
		 int n1, double* lat, int n2, double* lon, 
		 int n3, double *azimuth, int n4, double *zenith) 
    {
      if ((n1 != n2) || (n1 != n3) || (n1 != n4))
      {
        PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d,%d,%d) given",
		     n1, n2, n3, n4);
        return;
      }
      
      int i;
      cTime sutc;
      sutc.iYear    = yy; //int, year
      sutc.iMonth   = mm; //int, month
      sutc.iDay     = dd; //int, day    
      sutc.dHours   = h;
      sutc.dMinutes = m;
      sutc.dSeconds = s;
      
      cLocation udtLocation;
      cSunCoordinates ret;
      for (i=0; i<n1; i++)
      {
	udtLocation.dLatitude = lat[i];
	udtLocation.dLongitude = lon[i];
	sunpos(sutc, udtLocation, &ret);
	azimuth[i] = ret.dAzimuth;
	zenith[i] = ret.dZenithAngle;
      }
    }
%}




