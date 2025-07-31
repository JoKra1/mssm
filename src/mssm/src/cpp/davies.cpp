#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>

namespace py = pybind11;

/*
Small c++ rewrite of the c code to compute "Linear combination of chi-squared random variables" by Robert Davies avalilable at: https://www.robertnz.net/download.html.

Changes:
    - Interface with numpy so that the function can be called from python
    - No globals
    - No goto
*/

#define pi 3.14159265358979
#define log28 .0866  /*  log(2.0) / 8.0  */

double exp1(double x)
    { // Avoid unferflow
        if (x < -50)
            {
                return 0;
            }
        return exp(x);
    }

void counter(int &count, int lim)
    {
        /*  count number of calls to errbd, truncation, cfe */
        count += 1;
        if (count > lim)
            {
                throw 4;
            }
    }

double log1(double x, bool first)
   /* if (first) log(1 + x) ; else  log(1 + x) - x */
   {
    if (fabs(x) > 0.1)
    {
       return (first ? log(1.0 + x) : (log(1.0 + x) - x));
    }
    else
    {
        double s, s1, term, y, k;
        y = x / (2.0 + x);
        term = 2.0 * pow(y,3);
        k = 3.0;
        s = (first ? 2.0 : - x) * y;
        y = pow(y,2);
        for (s1 = s + term / k; s1 != s; s1 = s + term / k)
        { k = k + 2.0; term = term * y; s = s1; }
        return s;
    }
  }

  void order(double *lb, int *th, bool &ndtsrt, int r)
  /* find order of absolute values of lb */
  {
      int j, k; double lj;
      bool skipk;
      for ( j=0; j<r; j++ )
      {
         lj = fabs(lb[j]);
         skipk = false;
         for (k = j-1; k>=0; k--)
         {
            if ( lj > fabs(lb[th[k]]) )
            {
                th[k + 1] = th[k];
            }  
            else
            {
                skipk = true;
                break;
            } break;
         }
         if ( ! skipk ) {
            k = -1;
         }
         th[k + 1] = j;
      }
      ndtsrt = false;
   }

double   errbd(double* cx, double *lb, double *nc, int *n,
               int &count, double u, double sigsq, int lim, int r)
   /*  find bound on tail probability using mgf, cutoff
      point returned to *cx */
   {
      double sum1, lj, ncj, x, y, xconst; int j, nj;
      counter(count,lim);
      xconst = u * sigsq;  sum1 = u * xconst;  u = 2.0 * u;
      for (j=r-1; j>=0; j--)
      {
         nj = n[j]; lj = lb[j]; ncj = nc[j];
         x = u * lj; y = 1.0 - x;
         xconst = xconst + lj * (ncj / y + nj) / y;
         sum1 = sum1 + ncj * pow(x / y,2)
            + nj * (pow(x,2) / y + log1(-x, false ));
      }
      *cx = xconst; return exp1(-0.5 * sum1);
   }

double  ctff(double* upn, double *lb, double *nc, int *n,
             int &count, double sigsq, double accx, double lmin,
             double lmax, double mean, int lim, int r)
   /*  find ctff so that p(qf > ctff) < accx  if (upn > 0,
       p(qf < ctff) < accx otherwise */
   {
      double u1, u2, u, rb, xconst, c1, c2;
      u2 = *upn;   u1 = 0.0;  c1 = mean;
      rb = 2.0 * ((u2 > 0.0) ? lmax : lmin);
      for (u = u2 / (1.0 + u2 * rb); errbd(&c2, lb, nc, n, count, u, sigsq, lim, r) > accx; 
         u = u2 / (1.0 + u2 * rb))
      {
         u1 = u2;  c1 = c2;  u2 = 2.0 * u2;
      }
      for (u = (c1 - mean) / (c2 - mean); u < 0.9;
         u = (c1 - mean) / (c2 - mean))
      {
         u = (u1 + u2) / 2.0;
         if (errbd(&xconst, lb, nc, n, count, u / (1.0 + u * rb), sigsq, lim, r) > accx)
            {  u1 = u; c1 = xconst;  }
         else
            {  u2 = u;  c2 = xconst; }
      }
      *upn = u2; return c2;
   }

double truncation(double *lb, double *nc, int *n, int &count,
                  double u, double tausq, double sigsq, int lim, int r)
   /* bound integration error due to truncation at u */
   {
      double sum1, sum2, prod1, prod2, prod3, lj, ncj,
             x, y, err1, err2;
      int j, nj, s;

      counter(count,lim);
      sum1  = 0.0; prod2 = 0.0;  prod3 = 0.0;  s = 0;
      sum2 = (sigsq + tausq) * pow(u,2); prod1 = 2.0 * sum2;
      u = 2.0 * u;
      for (j=0; j<r; j++ )
      {
         lj = lb[j];  ncj = nc[j]; nj = n[j];
         x = pow(u * lj,2);
         sum1 = sum1 + ncj * x / (1.0 + x);
         if (x > 1.0)
         {
            prod2 = prod2 + nj * log(x);
            prod3 = prod3 + nj * log1(x, true);
            s = s + nj;
         }
         else  prod1 = prod1 + nj * log1(x, true);
      }
      sum1 = 0.5 * sum1;
      prod2 = prod1 + prod2;  prod3 = prod1 + prod3;
      x = exp1(-sum1 - 0.25 * prod2) / pi;
      y = exp1(-sum1 - 0.25 * prod3) / pi;
      err1 =  ( s  ==  0 )  ? 1.0 : x * 2.0 / s;
      err2 =  ( prod3 > 1.0 )  ? 2.5 * y : 1.0;
      if (err2 < err1) err1 = err2;
      x = 0.5 * sum2;
      err2 =  ( x  <=  y )  ? 1.0  : y / x;
      return  ( err1 < err2 )  ? err1  :  err2;
   }

void findu(double *lb, double *nc, int *n, double* utx,
           int &count, double accx, double sigsq, int lim, int r)
   /*  find u such that truncation(u) < accx and truncation(u / 1.2) > accx */
   {
      double u, ut; int i;
      static double divis[]={2.0,1.4,1.2,1.1};
      ut = *utx; u = ut / 4.0;
      if ( truncation(lb, nc, n, count, u, 0.0, sigsq, lim, r) > accx )
      {
         for ( u = ut; truncation(lb, nc, n, count, u, 0.0, sigsq, lim, r) > accx; u = ut) ut = ut * 4.0;
      }
      else
      {
         ut = u;
         for ( u = u / 4.0; truncation(lb, nc, n, count, u, 0.0, sigsq, lim, r) <=  accx; u = u / 4.0 )
         ut = u;
      }
      for ( i=0;i<4;i++)
         { u = ut/divis[i]; if ( truncation(lb, nc, n, count, u, 0.0, sigsq, lim, r)  <=  accx )  ut = u; }
      *utx = ut;
   }

   void integrate(double *lb, double *nc, int *n, double &intl, double &ersm, int nterm,
                  double interv, double tausq, bool mainx, double sigsq, double c, int r)
   /*  carry out integration with nterm terms, at stepsize
      interv.  if (! mainx) multiply integrand by
         1.0-exp(-0.5*tausq*u^2) */
   {
      double inpi, u, sum1, sum2, sum3, x, y, z;
      int k, j, nj;
      inpi = interv / pi;
      for ( k = nterm; k>=0; k--)
      {
         u = (k + 0.5) * interv;
         sum1 = - 2.0 * u * c;  sum2 = fabs(sum1);
         sum3 = - 0.5 * sigsq * pow(u,2);
         for ( j = r-1; j>=0; j--)
         {
            nj = n[j];  x = 2.0 * lb[j] * u;  y = pow(x,2);
            sum3 = sum3 - 0.25 * nj * log1(y, true );
            y = nc[j] * x / (1.0 + y);
            z = nj * atan(x) + y;
            sum1 = sum1 + z;   sum2 = sum2 + fabs(z);
            sum3 = sum3 - 0.5 * x * y;
         }
         x = inpi * exp1(sum3) / u;
	 if ( !  mainx )
         x = x * (1.0 - exp1(-0.5 * tausq * pow(u,2)));
         sum1 = sin(0.5 * sum1) * x;  sum2 = 0.5 * sum2 * x;
         intl = intl + sum1; ersm = ersm + sum2;
      }
   }

double cfe(double *lb, double *nc, int *th, int *n, bool &ndtsrt,
           bool &fail, int &count, double x, int lim, int r)
   /*  coef of tausq in error when convergence factor of
      exp1(-0.5*tausq*u^2) is used when df is evaluated at x */
   {
      double axl, axl1, axl2, sxl, sum1, lj; int j, k, t;
      counter(count,lim);
      if (ndtsrt) order(lb,th,ndtsrt,r);
      axl = fabs(x);  sxl = (x>0.0) ? 1.0 : -1.0;  sum1 = 0.0;
      for ( j = r-1; j>=0; j-- )
      { t = th[j];
         if ( lb[t] * sxl > 0.0 )
         {
            lj = fabs(lb[t]);
            axl1 = axl - lj * (n[t] + nc[t]);  axl2 = lj / log28;
            if ( axl1 > axl2 )  axl = axl1  ; else
            {
               if ( axl > axl2 )  axl = axl2;
               sum1 = (axl - axl1) / lj;
               for ( k = j-1; k>=0; k--)
               sum1 = sum1 + (n[th[k]] + nc[th[k]]);
               break;
            }
         }
      }
    if (sum1 > 100.0)
    { fail = true; return 1.0; } else
    return pow(2.0,(sum1 / 4.0)) / (pi * pow(axl,2));
   }

double  qf(double* lb1, double* nc1, int* n1, double* trace,
           int* ifault, double acc, double sigma, double c1, int r1,int lim1)
 
 /*  distribution function of a linear combination of non-central
    chi-squared random variables :
 
 input:
    lb[j]            coefficient of j-th chi-squared variable
    nc[j]            non-centrality parameter
    n[j]             degrees of freedom
    j = 0, 2 ... r-1
    sigma            coefficient of standard normal variable
    c                point at which df is to be evaluated
    lim              maximum number of terms in integration
    acc              maximum error
 
 output:
    ifault = 1       required accuracy NOT achieved
             2       round-off error possibly significant
             3       invalid parameters
             4       unable to locate integration parameters
             5       out of memory
 
    trace[0]         absolute sum
    trace[1]         total number of integration terms
    trace[2]         number of integrations
    trace[3]         integration interval in final integration
    trace[4]         truncation point in initial integration
    trace[5]         s.d. of initial convergence factor
    trace[6]         cycles to locate integration parameters     */
 
 {
       int j, nj, nt, ntm;
       double acc1, almx, xlim, xnt, xntm;
       double utx, tausq, sd, intv, intv1, x, up, un, d1, d2, lj, ncj;
       
       /*
       Old globals:
       */
       double sigsq, lmax, lmin, mean, c;
       double intl, ersm;
       int count, r, lim;
       bool ndtsrt, fail;
       int *n,*th;
       double *lb,*nc;
       /*
       End old globals
       */

       double qfval;
       static int rats[]={1,2,4,8};
 
       r=r1; lim=lim1; c=c1;
       n=n1; lb=lb1; nc=nc1;
       for ( j = 0; j<7; j++ )  trace[j] = 0.0;
       *ifault = 0; count = 0;
       intl = 0.0; ersm = 0.0;
       qfval = -1.0; acc1 = acc; ndtsrt = true;  fail = false;
       xlim = (double)lim;
       th=(int*)malloc(r*(sizeof(int)));
       try
       {
            if (! th) {throw 5;}
            /* find mean, sd, max and min of lb,
            check that parameter values are valid */
            sigsq = pow(sigma,2); sd = sigsq;
            lmax = 0.0; lmin = 0.0; mean = 0.0;
            for (j=0; j<r; j++ )
            {
                nj = n[j];  lj = lb[j];  ncj = nc[j];
                if ( nj < 0  ||  ncj < 0.0 ) {throw 3;}
                sd  = sd  + pow(lj,2) * (2 * nj + 4.0 * ncj);
                mean = mean + lj * (nj + ncj);
                if (lmax < lj) lmax = lj ; else if (lmin > lj) lmin = lj;
            }
            if ( sd == 0.0  )
                {  qfval = (c > 0.0) ? 1.0 : 0.0; free((char*)th); trace[6] = (double)count; return qfval; }
            if ( lmin == 0.0 && lmax == 0.0 && sigma == 0.0 )
                { throw 3; }
            sd = sqrt(sd);
            almx = (lmax < - lmin) ? - lmin : lmax;
    
            /* starting values for findu, ctff */
            utx = 16.0 / sd;  up = 4.5 / sd;  un = - up;
            /* truncation point with no convergence factor */
            findu(lb, nc, n, &utx, count, .5 * acc1, sigsq, lim, r);
            /* does convergence factor help */
            if (c != 0.0  && (almx > 0.07 * sd))
            {
                tausq = .25 * acc1 / cfe(lb,nc,th,n,ndtsrt,fail,count,c,lim,r);
                if (fail) fail = false ;
                else if (truncation(lb, nc, n, count, utx, tausq, sigsq, lim, r) < .2 * acc1)
                {
                    sigsq = sigsq + tausq;
                    findu(lb, nc, n, &utx, count, .25 * acc1, sigsq, lim, r);
                    trace[5] = sqrt(tausq);
                }
            }
            trace[4] = utx;  acc1 = 0.5 * acc1;
    
            /* find RANGE of distribution, quit if outside this */
            do
            {
                d1 = ctff(&up, lb, nc, n, count, sigsq, acc1, lmin, lmax, mean, lim, r) - c;
                if (d1 < 0.0) { qfval = 1.0; free((char*)th); trace[6] = (double)count; return qfval;}
                d2 = c - ctff(&un, lb, nc, n, count, sigsq, acc1, lmin, lmax, mean, lim, r);
                if (d2 < 0.0) { qfval = 0.0; free((char*)th); trace[6] = (double)count; return qfval;}
                /* find integration interval */
                intv = 2.0 * pi / ((d1 > d2) ? d1 : d2);
                /* calculate number of terms required for main and
                auxillary integrations */
                xnt = utx / intv;  xntm = 3.0 / sqrt(acc1);
                if (xnt > xntm * 1.5)
                {
                    /* parameters for auxillary integration */
                    if (xntm > xlim) { throw 1; }
                    ntm = (int)floor(xntm+0.5);
                    intv1 = utx / ntm;  x = 2.0 * pi / intv1;

                    if (x > fabs(c))
                    {
                        /* calculate convergence factor */
                        tausq = .33 * acc1 / (1.1 * (cfe(lb,nc,th,n,ndtsrt,fail,count,c - x,lim,r) + cfe(lb,nc,th,n,ndtsrt,fail,count,c + x,lim,r)));
                    }

                    if (fail || x <= fabs(c))
                    {
                        break;
                    }

                    acc1 = .67 * acc1;
                    /* auxillary integration */
                    integrate(lb, nc, n, intl, ersm, ntm, intv1, tausq, false, sigsq, c, r);
                    xlim = xlim - xntm;  sigsq = sigsq + tausq;
                    trace[2] = trace[2] + 1; trace[1] = trace[1] + ntm + 1;
                    /* find truncation point with new convergence factor */
                    findu(lb, nc, n, &utx, count, .25 * acc1, sigsq, lim, r);  acc1 = 0.75 * acc1;
                }
            } while (xnt > xntm * 1.5);
            
            /* main integration */
            trace[3] = intv;
            if (xnt > xlim) { throw 1; }
            nt = (int)floor(xnt+0.5);
            integrate(lb, nc, n, intl, ersm, nt, intv, 0.0, true, sigsq, c, r);
            trace[2] = trace[2] + 1; trace[1] = trace[1] + nt + 1;
            qfval = 0.5 - intl;
            trace[0] = ersm;
    
            /* test whether round-off error could be significant */
            up=ersm; x = up + acc / 10.0;
            for (j=0;j<4;j++) { if (rats[j] * x == rats[j] * up) *ifault = 2; }
            
       }
       catch(int e)
       {
        *ifault = e;
       }
    
       /*Free memory and return*/
       free((char*)th);
       trace[6] = (double)count;
       return qfval;
 }

std::tuple<double,int,py::array_t<double>> daviesQF(py::array_t<double, py::array::f_style | py::array::forcecast> lambda,
                                                    py::array_t<double, py::array::f_style | py::array::forcecast> noncen,
                                                    py::array_t<int, py::array::f_style | py::array::forcecast> n,
                                                    double c, double sigma, double acc, int r, int lim) {
   double q;
   int fault;

   // Create trace buffer
   auto trace = py::array_t<double>(7);

   // Access buffers
   py::buffer_info lb_buf = lambda.request();
   py::buffer_info nc_buf = noncen.request();
   py::buffer_info n_buf = n.request();
   py::buffer_info tr_buf = trace.request();

   // Cast to pointers
   double *lb = static_cast<double *>(lb_buf.ptr);
   double *nc = static_cast<double *>(nc_buf.ptr);
   double *tr = static_cast<double *>(tr_buf.ptr);
   int *n1 = static_cast<int *>(n_buf.ptr);
   
   // Now compute p value
   q = qf(lb, nc, n1, tr, &fault, acc, sigma, c, r, lim);

   return std::make_tuple(q,fault,std::move(trace));
}

PYBIND11_MODULE(davies, m) {
   m.doc() = "cpp code to compute p value under generalized chi square distribution";
   m.def("daviesQF", &daviesQF, "Compute p value under generalized chi square distribution");
}