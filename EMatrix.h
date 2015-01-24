#ifndef _EMATRIX_H
#define _EMATRIX_H 1

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <cstdarg>

#include <complex>
#include <fstream>
#include <iostream>

namespace ematrix {

//! These constants are from the gcc 3.2 <cmath> file (overkill??)
namespace cnst {
	const  double E          = 2.7182818284590452353602874713526625L;  // e         
	const  double LOG2E      = 1.4426950408889634073599246810018922L;  // log_2 e   
	const  double LOG10E     = 0.4342944819032518276511289189166051L;  // log_10 e  
	const  double LN2        = 0.6931471805599453094172321214581766L;  // log_e 2   
	const  double LN10       = 2.3025850929940456840179914546843642L;  // log_e 10  
	const  double PI         = 3.1415926535897932384626433832795029L;  // pi        
	const  double PI_2       = 1.5707963267948966192313216916397514L;  // pi/2 		
	const  double PI_4       = 0.7853981633974483096156608458198757L;  // pi/4 		
	const  double ONE_PI     = 0.3183098861837906715377675267450287L;  // 1/pi 		
	const  double TWO_PI     = 0.6366197723675813430755350534900574L;  // 2/pi 		
	const  double TWO_SQRTPI = 1.1283791670955125738961589031215452L;  // 2/sqrt(pi)
	const  double SQRT2      = 1.4142135623730950488016887242096981L;  // sqrt(2)   
	const  double SQRT1_2    = 0.7071067811865475244008443621048490L;  // 1/sqrt(2) 
	
	const  double RTD        = 180.0L/PI;                              // Radians to Degrees
	const  double DTR        =   1.0L/RTD;                             // Degrees to Radians

	/*! WGS84 Earth constants:
	 *  Titterton, D.H. and Weston, J.L.; <i>Strapdown Inertial Navigation
	 *  Technology</i>; Peter Peregrinus Ltd.; London 1997; ISBN 0-86341-260-2; pg. 53.
	 */
	const  double REQ        = 6378137.0;                              // meters
	const  double rEQ        = 6356752.3142;                           // meters
	const  double f          = 1.0/298.257223563;
	const  double ECC        = 0.0818191908426;
	const  double ECC2       = ECC*ECC;
	const  double WS         = 7.292115E-05;                           // rad/sec
  
	const  double g          = 9.78032534;                             // m/s^2
	const  double k          = 0.00193185;                             // ??
}

//This was necessary for a bad compiler that did not support the ansi C++
//abs overload necessary for complex<T> math.
//template<class tData0> 
//tData0 abs(tData0 x) { return (x > 0)?x:-x; }

/*! Overloaded comma initialization
 *  Veldhuizen, Todd; "Techniques for Scientific C++;" Indiana University
 *  Technical Report #542, Version 0.4, August 2000, pp 43-45
 *  http://www.osl.iu.edu/~tveldhui/papers
 */
template< class T_numtype, class T_iterator >
class ListInit {
  ListInit(); // disable default ctor.
protected:
  T_iterator iter_;
public:
  inline ListInit(T_iterator iter) : iter_(iter) {} 

  inline ListInit<T_numtype, T_iterator > operator,(T_numtype x) {
    *iter_ = x;
    return ListInit<T_numtype, T_iterator >(iter_ + 1);
  }
};

template < class tData, int tRows, int tCols >
class Matrix
{
protected:
  //! Memory allocation method
  void matalloc (int iRowIndex, int iColIndex);
  //! Number of row and columns
  int iRows, iCols;
  //! Storage element, memalloc above allocates a set of pointers to pointers
  tData *ij[tRows];
  tData storage[tRows*tCols];
public:

  //! Virtual destructor, no need though
  virtual ~Matrix ();

  /*! Default constructor
   *  Usage: Matrix< float > X;
   */  
  Matrix ();

  /*! Array initialize contructor
   *  Usage: float x[2][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0}};
   *         Matrix< float > X(2,3,&x[0]);
   */        
  Matrix (tData* tArray);

  /*! Standard Arg initialize contructor
   *  Usage: Matrix< float, 1, 2 > X(2.81,3.14);
   *  or     Matrix< float, 1, 3 > X(2.81,3.14,FLT_MAX);
   *  where FLT_MAX signals the end of the initialization; the rest of the
   *  array is set to zero from the call to matalloc.
   */  
  Matrix (tData first, ...) 
  : iRows(tRows), iCols(tCols) {
      matalloc( iRows, iCols);
      va_list p;
      va_start(p,first);

      tData x = first;
      if(x == FLT_MAX) return;
      ij[0][0] = x;

      for(int i = 1;i<(iRows*iCols);i++) {
        x = va_arg(p,tData);
        if(x == FLT_MAX) break;
        ij[0][i] = x;
      }
      va_end(p);
  }

  /*! Copy constructor (not to be confused with the assignment operator)
   *  Usage: Matrix< float > X(2,3);
   *         Matrix< float > Y = X;
   */  
  Matrix (const Matrix< tData, tRows, tCols > & R);

  /*! Assignment operator (not to be confused with the copy constructor)
   *  Usage: Y = X - Z;
   */  
  inline const Matrix< tData, tRows, tCols > &operator = (const Matrix< tData, tRows, tCols > & R);

  /*! Overloaded comma initialization
   *  Usage: Matrix< REAL, 1,3 >;
   *         A = 0.1,0.2,0.3;
   *  See the declaration of ListInit above for reference.
   */  
  inline ListInit<tData, tData* > operator = (tData x) {
    ij[0][0] = x;
    return ListInit<tData, tData* >(&ij[0][0] + 1);
  }

  /*! Array initializer
   *  Usage: float x[2][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0}};
   *         X.load(&x[0]);
   */  
  void load(tData* tArray);

  //! Row operator returns the matrix row corresponding to iRowIndex
  inline tData * operator [] (int iRowIndex) const { return ij[iRowIndex]; }

  /*! Data access operator for Matlab and FORTRAN indexing ... From 1 to n
   *  Note this does not imply FORTRAN memory storage
   *  Usage: X(1,2) = 5;
   *         float  = X(4,5);
   *  Note this looks similar to the memory initialize construstor,
   *  but is not the same thing
   */  
  inline tData & operator () (int iRowIndex, int iColIndex) const;

  /*! Vector Data access operator for Matlab and FORTRAN indexing ... From 1 to n
   *  Usage: Matrix<> X(4,1); X(2) = 5; *X(2) is equivalent to X(2,1)
   *  Usage: Matrix<> X(3,1); x = X(2); *X(2) is equivalent to X(1,2)
   *  Note this looks similar to the memory initialize construstor,
   *  but is not the same thing
   */  
  inline tData & operator () (int iIndex) const;

  /*! Overloaded output stream operator <<
   *  Usage: log_file << A;
   */  
  friend
  std::ostream& operator << (std::ostream& s,const Matrix< tData, tRows, tCols >& A)
  {
    int old_precision = s.precision(6);

    //s << "Address: 0x" << (&A) << std::endl;
    if(!A.ij) return s;
    for (int i=0; i<A.iRows; i++)
    {
        for (int j=0; j<A.iCols; j++)
        {
	        //s.width(25);
          s << (A[i][j]) << '\t';
        }
        s << std::endl;
    }
    s.flush();
    s.precision(old_precision);     
    return s;
  }

  
  /// This is broken but works good enough
  friend
  std::ostream& octave (std::ostream& s,const Matrix< tData, tRows, tCols >& A)
  {
    if(!A.ij) return s;
    
    s << "# name: z" << std::endl;
    s << "# type: complex matrix" << std::endl;
    s << "# rows: " << A.iRows << std::endl;
    s << "# columns: " << A.iCols << std::endl;
    s << A;
    s.flush();
    return s;
  }

  /*! Get the storage pointer for the data in the matrix
   *  This is really only here for the friend functions
   *  Try not to use it
   *  Usage: tData * x = A.pIJ();
   */  
  inline tData *pIJ (void) const { return ij[0]; }

  /*! Get the number of rows in a matrix
   *  Usage: int i = A.rows();
   */  
  inline int rows (void) const { return iRows; }
  
  /*! Get the number of cols in a matrix
   *  Usage: int i = A.cols();
   */  
  inline int cols (void) const { return iCols; }

  /*! Boolean == operator
   *  Usage: if (A == B) ...
   */  
  bool operator == (Matrix< tData, tRows, tCols > & R);
  
  /*! Boolean != operator
   *  Usage: if (A != B) ...
   */  
  bool operator != (Matrix< tData, tRows, tCols > & R);

  /*! Unary + operator
   *  Usage: C = (+B); Just returns *this;
   */  
  Matrix< tData, tRows, tCols > operator + ();

  /*! Unary - operator
   *  Usage: C = (-A);
   */  
  Matrix< tData, tRows, tCols > operator - ();

  /*! Addition operator
   *  Usage: C = A + B;
   */  
  Matrix< tData, tRows, tCols > operator + (const Matrix< tData, tRows, tCols > & R);

  /*! Subtaction operator
   *  Usage: C = A - B;
   */  
  Matrix< tData, tRows, tCols > operator - (const Matrix< tData, tRows, tCols > & R);

  /*! Scalar multiplication operator
   *  Usage: C = A * scalar;
   */  
  Matrix< tData, tRows, tCols > operator * (const tData & scalar);

  /*! Friend scalar multiplication operator
   *  Usage: C = scalar * A;
   */  
  friend
  Matrix< tData, tRows, tCols > operator * (const tData & scalar,const Matrix< tData, tRows, tCols > & R)
  {
    int iRows = R.iRows;
    int iCols = R.iCols;
    Matrix< tData, tRows, tCols >  Result = R;
    tData * pResult = Result.ij[0];
    for (int iIndex = 0; iIndex < iRows * iCols; iIndex++)
        (*pResult++) *= scalar;
    return Result;
  }

  /*! Matrix multiplication operator
   *  Usage: C = A * B;
   */  
  template < int tColsR >
  Matrix< tData, tRows, tColsR > operator * (const Matrix< tData, tCols, tColsR >& R)
  {
    tData x;
    Matrix< tData, tRows, tColsR > Result;

    for (int iIndex=0; iIndex<iRows; iIndex++)
    {
      for (int jIndex=0; jIndex<R.cols(); jIndex++)
      {
        x = tData(0);
        for (int kIndex=0; kIndex<R.rows(); kIndex++)
        {
          x += ij[iIndex][kIndex] * R[kIndex][jIndex];
        }
        Result[iIndex][jIndex] = x;
      }
    }
    return Result;
  }

  // Array multiplication operator
  // Usage: C = (A *= B); Must use parenthesis
  // This mimics Matlab's A .* B operator and not C's x *= 5 operator
  // Matrix< tData, tRows, tCols > operator *= (const Matrix< tData, tRows, tCols >  & R);

  // Array division operator
  // Usage: C = (A /= B); Must use parenthesis
  // This mimics Matlab's A ./ B operator and not C's x /= 5 operator
  // Matrix< tData, tRows, tCols > operator /= (const Matrix< tData, tRows, tCols >  & R);

  /*! Concatenate matrices top and botton
   *  Usage: C = (A & B); Must use parenthesis
   */  
  template < class tData0, int tRowsT, int tCols0, int tRowsB>
  friend Matrix< tData0, tRowsT+tRowsB, tCols0 > operator & (const Matrix< tData0, tRowsT, tCols0 >& Top,
                                                             const Matrix< tData0, tRowsB, tCols0 >& Bottom);

  /*! Concatenate matrices Left to Right
   *  Usage: C = (A | B); Must use parenthesis
   */  
  template < class tData0, int tRows0, int tColsL, int tColsR >
  friend Matrix< tData0, tRows0, tColsL+tColsR > operator | (const Matrix< tData0, tRows0, tColsL >& Left,
                                                             const Matrix< tData0, tRows0, tColsR >& Right);                                                     

  /*! Set contents to 0x0
   *  Usage: A.zeros();
   */  
  Matrix< tData, tRows, tCols > zeros( void );

  /*! Set contents to tData(1)
   *  Usage: A.ones();
   */  
  Matrix< tData, tRows, tCols > ones( void );

  /*! Set contents to the identity matrix
   *  Usage: A.eye();
   */  
  Matrix< tData, tRows, tRows > eye( void );

  /*! Matrix inverse, must link with Lapack
   *  Usage: A_inv = inv(A);
   */  
  template < int tRows0 > 
  friend Matrix< float, tRows0, tRows0 > inv( const Matrix< float, tRows0, tRows0 >& R );

  template < int tRows0 > 
  friend Matrix< std::complex<float>, tRows0, tRows0 > inv( const Matrix< std::complex<float>, tRows0, tRows0 >& R );

  template < int tRows0 > 
  friend Matrix< double, tRows0, tRows0 > inv( const Matrix< double, tRows0, tRows0 >& R );

  template < int tRows0 > 
  friend Matrix< std::complex<double>, tRows0, tRows0 > inv( const Matrix< std::complex<double>, tRows0, tRows0 >& R );

  /*! Matrix determinent, must link with Lapack
   *  Usage: A_det = det(A);
   */  
  template < int tRows0 > 
  friend float det( const Matrix< float, tRows0, tRows0 >& R );

  template < int tRows0 > 
  friend std::complex<float> det( const Matrix< std::complex<float>, tRows0, tRows0 >& R );

  template < int tRows0 > 
  friend double det( const Matrix< double, tRows0, tRows0 >& R );

  template < int tRows0 > 
  friend std::complex<double> det( const Matrix< std::complex<double>, tRows0, tRows0 >& R );

  // Matrix transpose and complex conjugate transpose
  // Usage: A_trans = trans(A);
  template < class tData0, int tRows0, int tCols0 > 
  friend Matrix< tData0, tCols0, tRows0 > trans( const Matrix< tData0, tRows0, tCols0 >& R );

  template < int tRows0, int tCols0 > 
  friend Matrix< std::complex<float>, tCols0, tRows0 > trans( const Matrix< std::complex<float> , tRows0, tCols0 >& R );

  template < int tRows0, int tCols0 > 
  friend Matrix< std::complex<double>, tCols0, tRows0 > trans( const Matrix< std::complex<double> , tRows0, tCols0 >& R );

  /*! Matrix diagonal like Matlab
   * This friend function does not modify input contents.
   * Usage: A_diag = diag(A);
   */  
  template < class tData0, int tRows0 > 
  friend Matrix< tData0, tRows0, 1 > diag( const Matrix< tData0, tRows0, tRows0 >& R );

  template < class tData0, int tRows0 > 
  friend Matrix< tData0, tRows0, tRows0 > diag( const Matrix< tData0, tRows0, 1 >& R );

  // Construct a skew symmetric matrix from a 3x1 vector.
  // w = [wx;wy;wz]
  // skew(w) = [0.0 -wz +wy
  //            +wz 0.0 -wx
  //            -wy +wx 0.0]
  // Usage: omega_x = skew(w);
  // friend Matrix< tData, 3, 3 > skew( const Matrix< tData, 3, 1 >& R );

  // Take the cross product of two 3x1 vectors
  // Usage: a = cross(lsr,Vc);
  // friend Matrix< tData, 3, 1 > cross( const Matrix< tData, 3, 1 >& L, const Matrix< tData, 3, 1 >& R );

  // Take the dot product of two 3x1 vectors
  // Usage: (norm(x))^2 = dot(x,x);
  // friend tData dot( const Matrix< tData, tRows, 1 >& L, const Matrix< tData, tRows, 1 >& R );
  
  /*! Take the norm of two vectors
   *  Usage: norm_a = a.n();
   */  
  tData n( void ) const;

  // Usage: norm_a = norm(a);
  // friend tData norm( const Matrix< tData, tRows, 1 >& R );

  /*! return a unit vector in the direction of V
   *  Usage: u_v = V.u()
   */  
  Matrix< tData, tRows, 1 > u( void ) const;

  /*! make all elements of the current matrix random from 0.0 to 1.0;
   *  Usage: u.rand();
   */  
  void rand(void);

  //friend Matrix< tData, 3, 3 > R(tData angle, char axis);

  //friend Matrix< tData, 3, 3 > TIE(tData t);

  //friend Matrix< tData, 3, 3 >  TEL(tData lat_rad,tData lon_rad);

  //friend Matrix< tData, 3, 3 >  ang_to_tib(tData psi,tData the,tData phi);

  //friend void uv_to_ae(tData *az,tData *el,Matrix< tData, 3, 1 > &u);

  void save( const char* filename, bool pgm );

};

template < class tData, int tRows, int tCols > 
void Matrix< tData, tRows, tCols >::save( const char* filename, bool pgm ) {
    int count = iRows*iCols;;
    std::ofstream ofs;
    
    //if(append) ofs.open(filename,IOS_BASE::app);
    //else ofs.open(filename);
    //if(!ofs) THROW_xBaseMatrixError;    
    
    std::string fname(filename);
    std::string closing(" ");
    std::ostringstream oss(std::ostringstream::out);
    
    if(pgm) {
      oss << "P2\n" << iRows << " " << iCols << "\n" << 255 << "\n";
      fname = fname + ".pgm";
      //closing += "0 0 ";
    } else {
      oss << iRows << " " << iCols << "\n";
    }
    
    ofs.open(fname.c_str());
    if(!ofs.good()) abort();
    
    ofs << oss.str();

    tData *pThis = ij[0];

    while(count) {
      ofs << *pThis << closing;
      pThis++;
      count--;
      if(!(count%iCols)) ofs << std::endl;
    } 
    ofs << std::endl;   
    ofs.close();
    return;
}

template < class tData, int tRows, int tCols >
void Matrix< tData, tRows, tCols >::matalloc(int iRowIndex, int iColIndex)
{
    if((iRowIndex*iColIndex) != 0) {
      ij[0] = &storage[0];
      for (int iIndex = 1; iIndex < iRowIndex; iIndex++)
         ij[iIndex] = ij[iIndex - 1] + iColIndex;

      std::memset (storage, 0x0, sizeof(storage));
    }
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols >::~Matrix()
{ }


template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols >::Matrix() 
: iRows(tRows), iCols(tCols)
{ 
    matalloc(tRows,tCols); 
}


template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols >::Matrix (tData* tArray)
: iRows (tRows), iCols (tCols)
{
    matalloc(tRows,tCols);
    std::memcpy(storage, tArray, sizeof(storage));
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols >::Matrix(const Matrix & R)
: iRows (R.iRows), iCols (R.iCols)
{
    if((iRows * iCols) != 0) {
      matalloc(R.iRows, R.iCols);
      std::memcpy(ij[0], R.ij[0], sizeof (tData) * iRows * iCols);
    }
}

template < class tData, int tRows, int tCols >
inline const Matrix< tData, tRows, tCols > & Matrix< tData, tRows, tCols >::operator = (const Matrix< tData, tRows, tCols > & R)
{
    if( this != &R )
    {
      if((iRows != R.iRows) || (iCols != R.iCols))  abort();
      std::memcpy (ij[0], R.ij[0], sizeof (tData) * iRows * iCols);
    }
    return *this;
}

template < class tData, int tRows, int tCols >
void Matrix< tData, tRows, tCols >::load(tData* tArray)
{
    std::memcpy(storage, tArray, sizeof(storage));
}

template < class tData, int tRows, int tCols >
inline tData & Matrix< tData, tRows, tCols >::operator () (int iRowIndex, int iColIndex) const 
{ 
    iRowIndex--;iColIndex--;
    if( iRowIndex<0 || iRows<=iRowIndex) abort();
    if( iColIndex<0 || iCols<=iColIndex) abort();
    return ij[iRowIndex][iColIndex];
}

template < class tData, int tRows, int tCols >
inline tData & Matrix< tData, tRows, tCols >::operator () (int iIndex) const 
{ 
  iIndex--;
  if( (iRows!=1) && (iCols!=1)) abort();
  if( iCols == 1 && (iIndex<0 || iRows<=iIndex)) abort();
  if( iRows == 1 && (iIndex<0 || iCols<=iIndex)) abort();
  return ij[0][iIndex];
}

template < class tData, int tRows, int tCols >
bool Matrix< tData, tRows, tCols >::operator == (Matrix< tData, tRows, tCols > & R)
{
  tData * pLeft  = ij[0];
  tData * pRight = R.ij[0];

  for (int iIndex = 0; iIndex < iRows * iCols; iIndex++)
    if((*pLeft++) != tData(*pRight++)) { 
      return false;
    }

  return true;
}

template < class tData, int tRows, int tCols >
bool Matrix< tData, tRows, tCols >::operator != (Matrix< tData, tRows, tCols > & R)
{
   return !(*this == R);
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator + ()
{
   return *this;
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator - ()
{
   Matrix< tData, tRows, tCols > Result;
   tData * pLeft = ij[0];
   tData * pResult = Result.ij[0];
   for (int iIndex = 0; iIndex < iRows * iCols; iIndex++)
      (*pResult++) = (-(*pLeft++));
   return Result;
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator + (const Matrix< tData, tRows, tCols > & R)
{
    Matrix< tData, tRows, tCols > Result;
    tData * pLeft   = ij[0];
    tData * pRight  = R.ij[0];
    tData * pResult = Result.ij[0];

    for (int iIndex = 0; iIndex < iRows * iCols; iIndex++)
      (*pResult++) = (*pLeft++) + (*pRight++);

    return Result;
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator - (const Matrix< tData, tRows, tCols > & R)
{
  Matrix< tData, tRows, tCols >  Result;
  tData * pLeft   = ij[0];
  tData * pRight  = R.ij[0];
  tData * pResult = Result.ij[0];

  for (int iIndex = 0; iIndex < iRows * iCols; iIndex++)
    (*pResult++) = (*pLeft++) - (*pRight++);
  return Result;
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator * (const tData &scalar)
{
   Matrix< tData, tRows, tCols >  Result = (*this);
   tData * pResult = Result.ij[0];
   for (int iIndex = 0; iIndex < iRows * iCols; iIndex++)
      (*pResult++) *= scalar;
   return Result;
}

// Array Math
//   template < class tData, int tRows, int tCols >
//   Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator *= (const Matrix< tData, tRows, tCols > & R)
//   {
//       Matrix< tData, tRows, tCols > Result;
//       tData  * pLeft   = ij[0];
//       tData * pRight  = R.ij[0];
//       tData  * pResult = Result.ij[0];
//       for (int iIndex = 0; iIndex < iRows * iCols; iIndex++)
//         (*pResult++) = (*pLeft++) * tData(*pRight++);
//       return Result;
//   }
// 
// 
//   template < class tData, int tRows, int tCols >
//   Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::operator /= (const Matrix< tData, tRows, tCols > & R)
//   {
//       Matrix< tData, tRows, tCols > Result;
//       tData * pLeft   = ij[0];
//       tData * pRight  = R.ij[0];
//       tData * pResult = Result.ij[0];
//       for (int iIndex = 0; iIndex < iRows * iCols; iIndex++)
//       {
//         if( *pRight == 0.0 )  abort();
//         (*pResult++) = (*pLeft++) / tData(*pRight++);
//       }
//       return Result;
//   }

template < class tData, int tRowsT, int tCols, int tRowsB >
Matrix< tData, tRowsT+tRowsB, tCols > operator & (const Matrix< tData, tRowsT, tCols >& Top,
                                                   const Matrix< tData, tRowsB, tCols >& Bottom)
{
    tData x;
    int iIndex,jIndex,kIndex,lIndex;

    Matrix< tData, tRowsT+tRowsB, tCols > Result;

    int i,ii,j;
    for(i=0;i<Top.iRows;i++)
      for(j=0;j<Top.iCols;j++)
         Result.ij[i][j] = Top.ij[i][j];

    for(i=Top.iRows,ii=0;i<(Top.iRows+Bottom.iRows);i++,ii++) {
      for(j=0;j<Top.iCols;j++) {
         Result.ij[i][j] = Bottom.ij[ii][j];
      }
    }
    return Result;
}

template < class tData, int tRows, int tColsL, int tColsR >
Matrix< tData, tRows, tColsL+tColsR > operator | (const Matrix< tData, tRows, tColsL >& Left,
                                                   const Matrix< tData, tRows, tColsR >& Right)                                                     
{
  Matrix< tData, tRows, tColsL+tColsR > Result;
  int i,j,jj;
  for(i=0;i<Left.iRows;i++)
    for(j=0;j<Left.iCols;j++)
       Result.ij[i][j] = Left.ij[i][j];
  for(i=0;i<Left.iRows;i++)
  {
    for(j=Left.iCols,jj=0;j<(Left.iCols+Right.iCols);j++,jj++)
    {
       Result.ij[i][j] = Right.ij[i][jj];
    }
  }
  return Result;
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::zeros( void )
{
  if(ij) std::memset (ij[0], 0x0, sizeof(tData) * iRows * iCols);
  return *this;
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tCols > Matrix< tData, tRows, tCols >::ones( void )
{
  tData * pThis = ij[0];
  if(iRows*iCols) {
      for (int iIndex = 0; iIndex < iRows * iCols; iIndex++)
        (*pThis++) = tData(1);
  }
  return *this;
}

template < class tData, int tRows, int tCols >
Matrix< tData, tRows, tRows > Matrix< tData, tRows, tCols >::eye( void )
{
  tData * pThis = ij[0];
  for (int iIndex = 0; iIndex < iRows; iIndex++, pThis+=iCols)
    (*pThis++) = tData(1);
  return *this;
}  

/*! non-portable declarations for Lapack
 *  Tested under i686 Redhat Linux
 */
extern "C" void sgesv_( const int &n, const int &nrhs, float *A,  
  const int &lda, int* ipiv, float *B, const int &ldb, int *info);
extern "C" void cgesv_( const int &n, const int &nrhs, std::complex<float> *A,  
  const int &lda, int* ipiv, std::complex<float> *B, const int &ldb, int *info);
extern "C" void dgesv_( const int &n, const int &nrhs, double *A,  
  const int &lda, int* ipiv, double *B, const int &ldb, int *info);
extern "C" void zgesv_( const int &n, const int &nrhs, std::complex<double> *A,  
  const int &lda, int* ipiv, std::complex<double> *B, const int &ldb, int *info);

template < int tRows > 
Matrix< float, tRows, tRows > inv( const Matrix< float, tRows, tRows >& R ) {
  int n = tRows;
  Matrix< float, tRows, tRows > a = R;
  int ipiv[tRows] = {0};
  Matrix< float, tRows, tRows > Result; 
  Result.eye(); 
  int info = 0;

  sgesv_(n, n, a.pIJ(), n, ipiv, Result.pIJ(), n, &info);

  if(info != 0) {
    std::cerr << "?gesv returned error: " << info << std::endl;
    abort();
  }  
  
  return Result;
}

template < int tRows > 
Matrix< std::complex<float>, tRows, tRows > inv( const Matrix< std::complex<float>, tRows, tRows >& R ) {
  int n = tRows;
  Matrix< std::complex<float>, tRows, tRows > a = R;
  int ipiv[tRows] = {0};
  Matrix< std::complex<float>, tRows, tRows > Result; 
  Result.eye(); 
  int info = 0;

  cgesv_(n, n, a.pIJ(), n, ipiv, Result.pIJ(), n, &info);

  if(info != 0) {
    std::cerr << "?gesv returned error: " << info << std::endl;
    abort();
  }  
  
  return Result;
}

template < int tRows > 
Matrix< double, tRows, tRows > inv( const Matrix< double, tRows, tRows >& R ) {
  int n = tRows;
  Matrix< double, tRows, tRows > a = R;
  int ipiv[tRows] = {0};
  Matrix< double, tRows, tRows > Result; 
  Result.eye(); 
  int info = 0;

  dgesv_(n, n, a.pIJ(), n, ipiv, Result.pIJ(), n, &info);

  if(info != 0) {
    std::cerr << "?gesv returned error: " << info << std::endl;
    abort();
  }  
  
  return Result;
}

template < int tRows > 
Matrix< std::complex<double>, tRows, tRows > inv( const Matrix< std::complex<double>, tRows, tRows >& R ) {
  int n           = tRows;
  Matrix< std::complex<double>, tRows, tRows > a = R;
  int ipiv[tRows] = {0};
  Matrix< std::complex<double>, tRows, tRows > Result; 
  Result.eye(); 
  int info        = 0;

  zgesv_(n, n, a.pIJ(), n, ipiv, Result.pIJ(), n, &info);

  if(info != 0) {
    std::cerr << "?gesv returned error: " << info << std::endl;
    abort();
  }  
  
  return Result;
}

/*! non-portable declarations for Lapack
 *  Tested under i686 Redhat Linux
 */
extern "C" void sgetrf_(const int &m, const int &n, float *A, 
  const int &lda, int *ipiv, int *info);
extern "C" void cgetrf_(const int &m, const int &n, std::complex<float> *A, 
  const int &lda, int *ipiv, int *info);
extern "C" void dgetrf_(const int &m, const int &n, double *A, 
  const int &lda, int *ipiv, int *info);
extern "C" void zgetrf_(const int &m, const int &n, std::complex<double> *A, 
  const int &lda, int *ipiv, int *info);

template < int tRows > 
float det( const Matrix< float, tRows, tRows >& R ) {
  int n = tRows;
  Matrix< float, tRows, tRows > a = R; 
  int ipiv[tRows] = {0};
  int info = 0;
  float result = 1.0; 

  sgetrf_(n,n,a.pIJ(),n,&ipiv[0],&info);

  if (info == 0) {
    for(int i=0;i<n;i++) {
      if(ipiv[i] != (i+1)) result *= -a[i][i]; // i+1 for fortran
      else                 result *= +a[i][i];
    }
  } else {
    std::cerr << "?getrf returned error: " << info << std::endl;
    abort();
  }
  return result;
}

template < int tRows > 
std::complex<float> det( const Matrix< std::complex<float>, tRows, tRows >& R ) {
  int n = tRows;
  Matrix< std::complex<float>, tRows, tRows > a = R; 
  int ipiv[tRows] = {0};
  int info = 0;
  std::complex<float> result(1.0,0.0); 

  cgetrf_(n,n,a.pIJ(),n,&ipiv[0],&info);

  if (info == 0) {
    for(int i=0;i<n;i++) {
      if(ipiv[i] != (i+1)) result *= -a[i][i]; // i+1 for fortran
      else                 result *= +a[i][i];
    }
  } else {
    std::cerr << "?getrf returned error: " << info << std::endl;
    abort();
  }
  return result;
}

template < int tRows > 
double det( const Matrix< double, tRows, tRows >& R ) {
  int n = tRows;
  Matrix< double, tRows, tRows > a = R; 
  int ipiv[tRows] = {0};
  int info = 0;
  double result = 1.0; 

  dgetrf_(n,n,a.pIJ(),n,&ipiv[0],&info);

  if (info == 0) {
    for(int i=0;i<n;i++) {
      if(ipiv[i] != (i+1)) result *= -a[i][i]; // i+1 for fortran
      else                 result *= +a[i][i];
    }
  } else {
    std::cerr << "?getrf returned error: " << info << std::endl;
    abort();
  }
  return result;
}

template < int tRows > 
std::complex<double> det( const Matrix< std::complex<double>, tRows, tRows >& R ) {
  int n = tRows;
  Matrix< std::complex<double>, tRows, tRows > a = R; 
  int ipiv[tRows] = {0};
  int info = 0;
  std::complex<double> result(1.0,0.0); 

  zgetrf_(n,n,a.pIJ(),n,&ipiv[0],&info);

  if (info == 0) {
    for(int i=0;i<n;i++) {
      if(ipiv[i] != (i+1)) result *= -a[i][i]; // i+1 for fortran
      else                 result *= +a[i][i];
    }
  } else {
    std::cerr << "?getrf returned error: " << info << std::endl;
    abort();
  }
  return result;
}

template < class tData, int tRows, int tCols > 
Matrix< tData, tCols, tRows > trans( const Matrix< tData, tRows, tCols >& R ) {
    Matrix< tData, tCols, tRows > Result;

    for (int iIndex = 0; iIndex < tCols; iIndex++)
      for (int jIndex = 0; jIndex < tRows; jIndex++)
        Result[iIndex][jIndex] = R[jIndex][iIndex];

    return Result;
}

template < int tRows, int tCols > 
Matrix< std::complex<float>, tCols, tRows > trans( const Matrix< std::complex<float> , tRows, tCols >& R ) {
    Matrix< std::complex<float>, tCols, tRows > Result;

    for (int iIndex = 0; iIndex < tCols; iIndex++)
      for (int jIndex = 0; jIndex < tRows; jIndex++)
        Result[iIndex][jIndex] = std::conj(R[jIndex][iIndex]);

    return Result;
}

template < int tRows, int tCols > 
Matrix< std::complex<double>, tCols, tRows > trans( const Matrix< std::complex<double> , tRows, tCols >& R ) {
    Matrix< std::complex<double>, tCols, tRows > Result;

    for (int iIndex = 0; iIndex < tCols; iIndex++)
      for (int jIndex = 0; jIndex < tRows; jIndex++)
        Result[iIndex][jIndex] = std::conj(R[jIndex][iIndex]);

    return Result;
}

template < class tData, int tRows > 
Matrix< tData, tRows, 1 > diag( const Matrix< tData, tRows, tRows >& R ) {
    Matrix< tData, tRows, 1 > Result;
    for (int iIndex = 0; iIndex < tRows; iIndex++ ) {
      Result[iIndex][0] = R[iIndex][iIndex];
    }
    return Result;
}

template < class tData, int tRows > 
Matrix< tData, tRows, tRows > diag( const Matrix< tData, tRows, 1 >& R ) {
    Matrix< tData, tRows, tRows > Result;
    for (int iIndex = 0; iIndex < tRows; iIndex++ ) {
      Result[iIndex][iIndex] = R[iIndex][0];
    }
    return Result;
}

template < class tData > 
Matrix< tData, 3, 3 > skew( const Matrix< tData, 3, 1 >& R ) {
  Matrix< tData, 3, 3 > Result;

  tData * pR  =  R.pIJ();

	Result(2,1) =  pR[2];
	Result(3,1) = -pR[1];
	Result(3,2) =  pR[0];

	Result(1,2) = -pR[2];
	Result(1,3) =  pR[1];
	Result(2,3) = -pR[0];

  return Result;
}
  
template < class tData > 
Matrix< tData, 3, 1 > cross( const Matrix< tData, 3, 1 >& L, const Matrix< tData, 3, 1 >& R ) {
    return skew(L)*R;
}

template < class tData, int tRows > 
tData dot( const Matrix< tData, tRows, 1 >& L, const Matrix< tData, tRows, 1 >& R ) {
  tData Result = tData(0);

  int irows = tRows;

  tData * pR = R.pIJ();
  tData * pL = L.pIJ();

  for (int iIndex = 0; iIndex < irows; iIndex++) {
    Result += (*pR++)*(*pL++);
  }
  return Result;
}

template < class tData, int tRows, int tCols > 
tData Matrix< tData, tRows, tCols >::n( void ) const {
    return std::sqrt(dot(*this,*this));
}

template < class tData, int tRows > 
tData norm( const Matrix< tData, tRows, 1 >& R ) {
  return std::sqrt(dot(R,R));
}

template < class tData, int tRows, int tCols > 
Matrix< tData, tRows, 1 > Matrix< tData, tRows, tCols >::u( void ) const {
    tData den = n();
    if(den == 0.0)  abort();
    Matrix< tData, tRows, 1 > Result = *this;
    return Result*(1.0/den);
}

template < class tData, int tRows, int tCols > 
void Matrix< tData, tRows, tCols >::rand(void) {
    for (int iIndex = 0; iIndex < iRows*iCols; iIndex++) {
      ij[0][iIndex] = std::rand()/tData(RAND_MAX);
    }
    return;
}

template < class tData > 
Matrix< tData, 3, 3 > R(tData angle, char axis) {
	if(axis == 'x') {
    tData result[3][3] = { {1.0,        0.0,         0.0},
		                       {0.0, cos(angle), -sin(angle)},
		                       {0.0, sin(angle),  cos(angle)},};
		return Matrix< tData, 3, 3 >((tData*)result);
	}
	else if(axis=='y') {
		tData result[3][3] = { { cos(angle), 0.0, sin(angle)},
		                      {       0.0,  1.0,        0.0},
		                      {-sin(angle), 0.0, cos(angle)},};
		return Matrix< tData, 3, 3 >((tData*)result);
	}
	else if(axis=='z') {
		tData result[3][3] = { {cos(angle), -sin(angle),  0.0},
		                      {sin(angle),  cos(angle),  0.0},
		                      {0.0,                0.0,  1.0},};
		return Matrix< tData, 3, 3 >((tData*)result);
	}
  else  {
    abort();
  }
}

template < class tData > 
Matrix< tData, 3, 3 > TIE(tData t) {
  return R(cnst::WS*t,'z');
}

template < class tData > 
Matrix< tData, 3, 3 >  TEL(tData lat_rad,tData lon_rad) {
  return (R(lon_rad,'z')*R(-lat_rad-cnst::PI_2,'y'));
}

template < class tData > 
Matrix< tData, 3, 3 >  ang_to_tib(tData psi,tData the,tData phi) {
  tData a_r_psi[] = {cos(psi),-sin(psi),        0,
                     sin(psi), cos(psi),        0,
                            0,        0,        1};
  Matrix< tData, 3, 3 > r_psi(a_r_psi);                        

  tData a_r_the[] = {cos(the),        0, sin(the),
                            0,        1,        0,
                    -sin(the),        0, cos(the)};
  Matrix< tData, 3, 3 > r_the(a_r_the);                        

  tData a_r_phi[] = {       1,        0,        0,       
                            0, cos(phi),-sin(phi),
                            0, sin(phi), cos(phi)};                         
  Matrix< tData, 3, 3 > r_phi(a_r_phi);                        

  return r_psi*r_the*r_phi;
}

template < class tData > 
void uv_to_ae(tData *az,tData *el,Matrix< tData, 3, 1 > &u) {
  if(norm(u) != 1.0) u=u*(1.0/norm(u));
  *az=atan2(u(2),u(1));
  *el=asin(-u(3));
  return;
}


} // from namespace

#endif // from _EMATRIX_H
