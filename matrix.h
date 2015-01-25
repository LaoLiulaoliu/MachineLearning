#ifndef _MATRIX_H
#define _MATRIX_H 1

#include <cstdlib>

#include <iostream>
#include <vector>

namespace matrix {

template < class T, size_t row, size_t col >
class Matrix
{
putlic:
    Matrix();
    virtual ~Matrix();

    inline size_t rows (void) const { return iRows; }

    inline size_t cols (void) const { return iCols; }

    inline T & operator () (int iRowIndex, int iColIndex) const;


    template < int colR >
    Matrix< T, row, col > operator * (const Matrix< T, col, colR >& R)
    {
        T x;
        Matrix< T, row, colR > result;

        for (size_t iIndex=0; iIndex<iRows; iIndex++)
        {
          for (size_t jIndex=0; jIndex<R.cols(); jIndex++)
          {
            x = tData(0);
            for (size_t kIndex=0; kIndex<R.rows(); kIndex++)
            {
              x += ij(iIndex, kIndex) * R(kIndex, jIndex);
            }
            result(iIndex, jIndex) = x;
          }
        }
        return result;
    }

private:
    vector< vector<T> > X2d;
    T iRows;
    T iCols;
};


template < class T, size_t row, size_t col >
inline T & Matrix< T, row, col >::operator () (int iRowIndex, int iColIndex) const
{
    if( iRowIndex<0 || iRows<=iRowIndex) abort();
    if( iColIndex<0 || iCols<=iColIndex) abort();
    return X2d[iRowIndex][iColIndex];
}


} // from namespace matrix

#endif // from _MATRIX_H
