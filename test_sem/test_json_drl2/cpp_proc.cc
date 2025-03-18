#include <iostream>
#include <semaphore.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    // Create semaphores
    sem_t *sem_cpp = sem_open("/sem_cpp_json_drl", O_CREAT, 0666, 0);  // C++ starts first
    sem_t *sem_py = sem_open("/sem_py_json_drl", O_CREAT, 0666, 0);

    for (int i = 0; i < 30; i++) {
        if (i > 0) {
            sem_wait(sem_cpp);  // Wait for Python to write to data2.json

            // Read from data2.json
            std::ifstream f2("data.json");
            json data2;
            f2 >> data2;
            f2.close();
            std::cout << "[C++] Received: " << data2["message"] << std::endl;
            std::cout << "[C++] Applied " << data2["message"] << " to Network" << std::endl;

            sem_post(sem_py);  // Signal Python for next iteration
        }
        std::cout << "[C++] Updating Network Simulator......!!!!!!"
    }

    // Cleanup
    sem_close(sem_cpp);
    sem_close(sem_py);
    sem_unlink("/sem_cpp_json_drl");
    sem_unlink("/sem_py_json_drl");
    return 0;
}
