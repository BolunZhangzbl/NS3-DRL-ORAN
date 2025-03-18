#include <iostream>
#include <semaphore.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    // Create semaphores: C++ starts first (sem1 = 1, sem2 = 0)
    sem_t *sem_cpp = sem_open("/sem_cpp", O_CREAT, 0666, 1);  // C++ controls this
    sem_t *sem_py = sem_open("/sem_py", O_CREAT, 0666, 0);    // Python waits on this

    if (sem_cpp == SEM_FAILED || sem_py == SEM_FAILED) {
        perror("sem_open failed");
        return 1;
    }

    std::cout << "[C++] Starting first! Signaling Python next..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "[C++] Waiting on sem_cpp..." << std::endl;
        sem_wait(sem_cpp);  // Wait on sem_cpp (starts at 1)
        std::cout << "[C++] Working... (Step " << (i + 1) << ")" << std::endl;
        sleep(1);
        std::cout << "[C++] Signaling Python via sem_py..." << std::endl;
        sem_post(sem_py);   // Signal Python via sem_py
    }

    sem_close(sem_cpp);
    sem_close(sem_py);
    sem_unlink("/sem_cpp");  // Cleanup
    sem_unlink("/sem_py");
    return 0;
}
