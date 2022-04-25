/**
 * @file CPPThreadTest.cpp
 * @author RoyenHeart
 * @brief C++ Thread Test and Practice
 * @version 0.1
 * @date 2022-03-15
 * 
 */
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <thread>
#include <assert.h>
#include <string>

using namespace std;

void t_hello() {
    cout << "Hello World With Thread!\n";
}

void t_transfromData(int i) {
    cout << "Data i is : " << i << endl;
}

int main(int argc, char *agrv) {
    // --Thread tHello-- //
    thread tHello(t_hello);
    cout << "tHello runs in id : " << tHello.get_id() << endl;
    cout << "Is this thread joinable? " << tHello.joinable() << endl;
    tHello.join();
    // --Thread tHello End-- //

    // --print maximum threads per process-- //
    cout << thread::hardware_concurrency() << endl;
    // --print maximum threads per process End-- //

    // --显示和隐式转换的影响-- //
    double transfromData = 3.444434;
    thread tTransfromData1(t_transfromData, transfromData);
    thread tTransfromData2(t_transfromData, (int)transfromData);
    tTransfromData1.join();
    tTransfromData2.join();
    // --显示和隐式转换的影响 End-- //

    return EXIT_SUCCESS;
}