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
std::vector<std::vector<double>> test_action;

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

    out.open("./res/action");
    for(unsigned int i = 0; i < N; i++)
        out << i << (i != N-1 ? "," : "\n");
    for(unsigned int t = 0; t < EXT-OBS; t++) {
        for(unsigned int i = 0; i < N; i++)
            out << test_action[i][t] << (i != N-1 ? "," : "\n");
    }
    out.close();

    std::system("./python/plot.py");
}

void clean() {
    std::vector<GBMParam>().swap(param);
    std::vector<std::vector<double>>().swap(path);
    std::vector<Memory>().swap(memory);
    std::vector<double>().swap(mean_reward);
    std::vector<double>().swap(test);
    std::vector<std::vector<double>>().swap(test_action);
}

std::vector<double> sample_state(unsigned int t) {
    std::vector<double> state(N);
    for(unsigned int i = 0; i < N; i++)
        state[i] = log(path[i][t]) - log(path[i][t-OBS]);
    return state;
}

int main(int argc, char *argv[])
{
    std::cout << std::fixed;
    std::cout.precision(12);

    param.resize(N);
    for(unsigned int i = 0; i < N; i++)
        param[i] = GBMParam(INITIAL_VALUE, MU, SIGMA);

    path = gbm(param, EXT, seed);

    Net actor;    
    actor.add_layer(N,  50);
    actor.add_layer(50, 50);
    actor.add_layer(50, 50);
    actor.add_layer(50, 50);
    actor.add_layer(50,  N);
    actor.use_softmax();
    actor.init(seed);

    Net critic;
    critic.add_layer(N+N, 50);
    critic.add_layer(50,  50);
    critic.add_layer(50,  50);
    critic.add_layer(50,  50);
    critic.add_layer(50,   1);
    critic.init(seed);

    DDPG ddpg(actor, critic);

    double eps = EPS_INIT;

    for(unsigned int itr = 0; itr < ITR; itr++) {
        unsigned int update_count = 0;
        double reward_sum = 0.00, q_sum = 0.00;
        for(unsigned int t = OBS; t < EXT; t++) {
            if((itr+1)*t > OBS && eps > EPS_MIN)
                eps += (EPS_MIN - EPS_INIT) / CAPACITY;
            std::vector<double> state = sample_state(t);
            std::vector<double> action = ddpg.epsilon_greedy(state, eps);
            std::vector<double> next_state = sample_state(t+1);

            double mean  = 0.00;
            double model = 0.00;
            for(unsigned int i = 0; i < N; i++) {
                mean  += path[i][t+1];
                model += path[i][t+1] * action[i];
            }
            mean /= N;

            double reward = log(model) - log(mean);
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

        mean_reward.push_back(reward_sum / (EXT-OBS));

        std::cout << "ITR=" << itr << " ";
        std::cout << "MR=" << mean_reward.back() << " ";
        std::cout << "Q=" << q_sum / update_count << "\n";
    }

    test_action.resize(N, std::vector<double>(EXT-OBS));

    for(unsigned int t = OBS; t < EXT; t++) {
        std::vector<double> state = sample_state(t);
        std::vector<double> action = ddpg.epsilon_greedy(state, 0.00);

        double reward = 0.00;
        for(unsigned int i = 0; i < N; i++) {
            reward += path[i][t+1] * action[i];
            test_action[i][t-OBS] = action[i];
        }
        test.push_back(reward);

        std::vector<double>().swap(state);
        std::vector<double>().swap(action);
    }

    write();
    clean();
    
    return 0;
}