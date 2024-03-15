#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

#include "../lib/gbm.hpp"

double GBMParam::p0() { return p; }
double GBMParam::mu() { return m; }
double GBMParam::sigma() { return s; }
double GBMParam::drift() { return d; }

void normal(std::vector<std::vector<double>> &dat, std::default_random_engine &seed) {
    std::normal_distribution<double> std_normal(0.0, 1.0);
    for(int i = 0; i < dat.size(); i++)
        for(int j = 0; j < dat[i].size(); j++) dat[i][j] = std_normal(seed);
}

void cumsum(std::vector<std::vector<double>> &dat) {
    for(int i = 0; i < dat.size(); i++)
        for(int j = 0; j < dat[i].size(); j++) dat[i][j] += dat[i][j-1];
}

std::vector<std::vector<double>> gbm(std::vector<GBMParam> &param, unsigned int ext, std::default_random_engine &seed) {
    std::vector<std::vector<double>> path(param.size(), std::vector<double>(ext+1));
    normal(path, seed); cumsum(path);
    for(unsigned int i = 0; i < param.size(); i++) {
        path[i][0] = param[i].p0();
        for(unsigned int j = 1; j <= ext; j++) {
            path[i][j] *= param[i].sigma();
            path[i][j] += param[i].drift() * j;
            path[i][j] = param[i].p0() * exp(path[i][j]);
        }
    }
    return path;
}