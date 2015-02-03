#ifndef _MATRIX_H
#define _MATRIX_H 1

#include <cstdlib>

#include <iostream>
#include <vector>

using namespace std;

namespace matrix {

template < class T, size_t row, size_t col >
class Matrix
{
public:
//    Matrix();
//    virtual ~Matrix();

    Matrix() {}
    virtual ~Matrix() {}
    Matrix (T first, ...)
    :iRows(row), iCols(col) {}

    Matrix(vector< vector<T> > &data) {
        iRows = data.size();
        iCols = data[0].size();
        Data.resize(iRows);

        for (size_t i = 0; i < iRows; ++i) {
            vector<T> vec(iCols);
            for (size_t j = 0; j < iCols; ++j) {
                vec.push_bach(data[i][j]);
            }
            Data.push_back(vec);
        }
    }

    inline size_t rows (void) const { return iRows; }

    inline size_t cols (void) const { return iCols; }

    inline T & operator () (int iRowIndex, int iColIndex) const;

    inline void set_element(T element, size_t iRowIndex, size_t iColIndex);

    void load(vector< vector<T> > &data);


    void print_data()
    {
        for (typename vector< vector<T> >::iterator i = Data.begin(); i != Data.end(); ++i) {
            for (typename vector<T>::iterator j = i->begin(); j != i->end(); ++j) {
                cout << *j << " ";
            }
            cout << endl;
        }
    }

    template < size_t colR >
    Matrix< T, row, col > operator * (const Matrix< T, col, colR >& R)
    {
        T x;
        Matrix< T, row, colR > result;

        for (size_t iIndex=0; iIndex<iRows; iIndex++)
        {
          for (size_t jIndex=0; jIndex<R.cols(); jIndex++)
          {
            x = T(0);
            for (size_t kIndex=0; kIndex<R.rows(); kIndex++)
            {
              x += Data(iIndex, kIndex) * R(kIndex, jIndex);
            }
            result(iIndex, jIndex) = x;
          }
        }
        return result;
    }

private:
    vector< vector<T> > Data;
    size_t iRows;
    size_t iCols;
};


template < class T, size_t row, size_t col >
inline T & Matrix< T, row, col >::operator () (int iRowIndex, int iColIndex) const
{
    if( iRowIndex<0 || iRows<=iRowIndex ) abort();
    if( iColIndex<0 || iCols<=iColIndex ) abort();
    return Data[iRowIndex][iColIndex];
}

template < class T, size_t row, size_t col >
inline void Matrix< T, row, col >::set_element(T element, size_t iRowIndex, size_t iColIndex)
{
    if( iRowIndex<0 || iRows<=iRowIndex ) abort();
    if( iColIndex<0 || iCols<=iColIndex ) abort();
    Data[iRowIndex][iColIndex] = element;
}

template < class T, size_t row, size_t col >
void Matrix< T, row, col >::load(vector< vector<T> > &data)
{
    iRows = data.size();
    iCols = data[0].size();
    Data.resize(iRows);

    for (size_t i = 0; i < iRows; ++i) {
        vector<T> vec(iCols);
        for (size_t j = 0; j < iCols; ++j) {
            vec.push_bach(data[i][j]);
        }
        Data.push_back(vec);
    }
}

} // from namespace matrix

#endif // from _MATRIX_H
