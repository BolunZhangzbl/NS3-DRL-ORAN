#include <iostream>
#include <semaphore.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <chrono>
#include <thread>

using json = nlohmann::json;
using namespace std::chrono;

int main() {
    // Create semaphores
    sem_t *sem_cpp = sem_open("/sem_cpp_json_drl", O_CREAT, 0666, 0);
    sem_t *sem_py = sem_open("/sem_py_json_drl", O_CREAT, 0666, 0);

    auto last_interaction_time = steady_clock::now(); // Track the last interaction time
    int step = 0;

    while (true) {
        // Constantly updating network simulator
        std::cout << "[C++] Step: " << step << " Updating Network Simulator......!!!!!!" << std::endl;
        std::this_thread::sleep_for(milliseconds(10));  // Simulating small network updates

        // Check if 100ms has passed for interaction with Python
        auto now = steady_clock::now();
        if (duration_cast<milliseconds>(now - last_interaction_time).count() >= 100) {
            last_interaction_time = now; // Reset interaction timer

            sem_wait(sem_cpp);  // Wait for Python to write data.json

            // Read from data.json
            std::ifstream f2("data.json");
            json data2;
            f2 >> data2;
            f2.close();
            std::cout << "[C++] Received: " << data2["message"] << std::endl;
            std::cout << "[C++] Applied " << data2["message"] << " to Network" << std::endl;

            sem_post(sem_py);  // Signal Python for next iteration
        }

        step++;  // Increment step counter
    }

    // Cleanup
    sem_close(sem_cpp);
    sem_close(sem_py);
    sem_unlink("/sem_cpp_json_drl");
    sem_unlink("/sem_py_json_drl");

    return 0;
}
