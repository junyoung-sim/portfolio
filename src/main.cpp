#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "../lib/param.hpp"
#include "../lib/gbm.hpp"
#include "../lib/ddpg.hpp"

std::ofstream out;
std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

std::vector<GBMParam> param;
std::vector<std::vector<double>> path;

std::vector<Memory> memory;

std::vector<double> mean_reward;
std::vector<double> test;

std::vector<double> sample_state(unsigned int t) {
    std::vector<double> state(N);
    for(unsigned int i = 0; i < N; i++)
        state[i] = (path[i][t] - path[i][t-1]) / path[i][t-1];
    return state;
}

void write() {
    out.open("./res/path");
    for(unsigned int i = 0; i < N; i++)
        out << i << (i != N-1 ? "," : "\n");
    for(unsigned int t = 0; t <= EXT; t++) {
        for(unsigned int i = 0; i < N; i++)
            out << path[i][t] << (i != N-1 ? "," : "\n");
    }
    out.close();

    out.open("./res/log");
    out << "mr\n";
    for(unsigned int i = 0; i < ITR; i++)
        out << mean_reward[i] << "\n";
    out.close();

    out.open("./res/test");
    out << "test\n";
    for(double &x: test)
        out << x << "\n";
    out.close();

    std::system("./python/plot.py");
}

void clean() {
    std::vector<GBMParam>().swap(param);
    std::vector<std::vector<double>>().swap(path);
    std::vector<Memory>().swap(memory);
    std::vector<double>().swap(mean_reward);
    std::vector<double>().swap(test);
}

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(6);

    param.resize(N);
    for(unsigned int i = 0; i < N; i++)
        param[i] = GBMParam(INITIAL_VALUE, MU, SIGMA);

    path = gbm(param, EXT, seed);

    Net actor;
    actor.add_layer(N+0, N+N);
    actor.add_layer(N+N, N+N);
    actor.add_layer(N+N, N+N);
    actor.add_layer(N+N, N+0);
    actor.use_softmax();
    actor.init(seed);

    Net critic;
    critic.add_layer(N+N, N+0);
    critic.add_layer(N+0, N+0);
    critic.add_layer(N+0, N+0);
    critic.add_layer(N+0, 1);
    critic.init(seed);

    DDPG ddpg(actor, critic);

    double eps = EPS_INIT;
    for(unsigned int itr = 0; itr < ITR; itr++) {
        unsigned int update_count = 0;
        double reward_sum = 0.00, q_sum = 0.00;
        for(unsigned int t = 1; t < EXT; t++) {
            if((itr+1)*t > 1 && eps > EPS_MIN)
                eps += (EPS_MIN - EPS_INIT) / CAPACITY;
            std::vector<double> state = sample_state(t);
            std::vector<double> action = ddpg.epsilon_greedy(state, eps);
            std::vector<double> next_state = sample_state(t+1);

            double reward = 0.00;
            for(unsigned int i = 0; i < N; i++)
                reward += path[i][t+1] * action[i];
            reward = log10(reward);
            reward_sum += reward;
            
            memory.push_back(Memory(state, action, next_state, reward));

            if(memory.size() == CAPACITY) {
                std::vector<unsigned int> index(CAPACITY, 0);
                std::iota(index.begin(), index.end(), 0);
                std::shuffle(index.begin(), index.end(), seed);
                index.erase(index.begin() + BATCH, index.end());

                for(unsigned int &k: index)
                    q_sum += ddpg.optimize(memory[k], GAMMA, ALPHA, LAMBDA);
                update_count += BATCH;

                memory.erase(memory.begin());
                std::vector<unsigned int>().swap(index);
            }
        }

        mean_reward.push_back(reward_sum / (EXT-1));

        std::cout << "ITR=" << itr << " ";
        std::cout << "MR=" << mean_reward.back() << " ";
        std::cout << "Q=" << q_sum / update_count << "\n";
    }

    for(unsigned int t = 1; t < EXT; t++) {
        std::vector<double> state = sample_state(t);
        std::vector<double> action = ddpg.epsilon_greedy(state, 0.00);

        double reward = 0.00;
        for(unsigned int i = 0; i < N; i++)
            reward += path[i][t+1] * action[i];
        test.push_back(reward);

        std::vector<double>().swap(state);
        std::vector<double>().swap(action);
    }

    write();
    clean();
    
    return 0;
}