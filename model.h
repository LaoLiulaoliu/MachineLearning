#include "matrix.h"

#include <fstream>

using namespace matrix;

template < class T >
class Model
{
public:
    Model();
    virtual ~Model();

    int read_data();
    int print_X();
    int print_Y();


private:

    vector< vector<T> > X2d;
    vector< vector<T> > Y2d;
    Matrix< T,1,1 > X;
    Matrix< T,1,1 > Y;
};


template <class T>
int Model<T>::read_data()
{
    T x1, x2, y;
    ifstream infile("data.txt");
    while (infile >> x1 >> x2 >> y) {
        vector<T> vec_x(2);
        vec_x[0] = x1;
        vec_x[1] = x2;
        X2d.push_back(vec_x);
        vector<T> vec_y(1);
        Y2d.push_back(vec_y);
    }

    return 0;
}

template <class T>
int Model<T>::print_X()
{
    for (typename vector< vector<T> >::iterator i = X2d.begin(); i != X2d.end(); ++i) {
        for (typename vector<T>::iterator j = i->begin(); j != i->end(); ++j) {
            cout << *j << " ";
        }
        cout << endl;
    }

    return 0;
}

template <class T>
int Model<T>::print_Y()
{
    for (size_t i = 0; i < Y2d.size(); ++i) {
        for (size_t j = 0; j < Y2d[i].size(); ++j) {
            cout << Y2d[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}

