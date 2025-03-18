#include <iostream>
#include <semaphore.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    // Create semaphores
    sem_t *sem_cpp = sem_open("/sem_cpp_json_drl", O_CREAT, 0666, 1);  // C++ starts first
    sem_t *sem_py = sem_open("/sem_py_json_drl", O_CREAT, 0666, 0);

    for (int i = 0; i < 30; ++i) {
        sem_wait(sem_cpp);  // Wait for turn (starts at 1)

        // Write to data1.json
        json data1 = {{"message", "[C++] Hello Python!"}};
        std::ofstream f1("data1.json");
        f1 << data1.dump(4);  // Pretty-print with 4 spaces
        f1.close();

        sem_post(sem_py);  // Signal Python to read data1.json

        sem_wait(sem_cpp);  // Wait for Python to write to data2.json

        // Read from data2.json
        std::ifstream f2("data2.json");
        json data2;
        f2 >> data2;
        f2.close();
        std::cout << "[C++] Received: " << data2["message"] << std::endl;

        sem_post(sem_py);  // Signal Python for next iteration
    }

    // Cleanup
    sem_close(sem_cpp);
    sem_close(sem_py);
    sem_unlink("/sem_cpp_json_drl");
    sem_unlink("/sem_py_json_drl");
    return 0;
}


//int main() {
//    // Create semaphores: sem_cpp starts at 1, sem_py at 0
//    sem_t *sem_cpp = sem_open("/sem_cpp_json_drl", O_CREAT, 0666, 1);
//    sem_t *sem_py = sem_open("/sem_py_json_drl", O_CREAT, 0666, 0);
//
//    if (sem_cpp == SEM_FAILED || sem_py == SEM_FAILED) {
//        std::cerr << "Error opening semaphores!" << std::endl;
//        return 1;
//    }
//
//    // Loop for 30 iterations
//    for (int i = 0; i < 30; ++i) {
//        if (i % 10 == 0) {
//            // Wait for Python to be ready (sem_cpp starts at 1, so it waits here)
//            sem_wait(sem_cpp);
//
//            // Prepare and write data1.json
//            json data1 = {{"message", "[C++] Hello Python!"}};
//            std::ofstream f1("data1.json");
//            f1 << data1.dump(4);  // Pretty-print with 4 spaces
//            f1.close();
//
//            // Signal Python to read data1.json
//            sem_post(sem_py);
//        }
//
//        // Wait for Python to write to data2.json
//        sem_wait(sem_cpp);
//
//        // Read from data2.json
//        json data2;
//        std::ifstream f2("data2.json");
//        f2 >> data2;
//        f2.close();
//        std::cout << "[C++] Received: " << data2["message"] << std::endl;
//        std::cout << "[C++] Applied " << data2["message"] << " to Network" << std::endl;
//
//        // Skip the data1.json write in iteration 0
//        if (i % 10 != 0) {
//            // Write to data1.json in the other iterations
//            json data1 = {{"message", "[C++] Hello Python!"}};
//            std::ofstream f1("data1.json");
//            f1 << data1.dump(4);
//            f1.close();
//        }
//
//        // Signal Python for next iteration
//        sem_post(sem_py);
//    }
//
//    // Cleanup
//    sem_close(sem_cpp);
//    sem_close(sem_py);
//    sem_unlink("/sem_cpp_json_drl");
//    sem_unlink("/sem_py_json_drl");
//
//    return 0;
//}
