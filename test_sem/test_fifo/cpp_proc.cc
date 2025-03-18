#include <iostream>
#include <semaphore.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

int main() {
    // Create FIFOs (named pipes)
    mkfifo("./fifo1", 0666);  // C++ writes to this
    mkfifo("./fifo2", 0666);  // Python writes to this

    // Create semaphores
    sem_t *sem_cpp = sem_open("/sem_cpp_fifo", O_CREAT, 0666, 1);  // C++ starts first
    sem_t *sem_py = sem_open("/sem_py_fifo", O_CREAT, 0666, 0);

    for (int i = 0; i < 3; ++i) {
        sem_wait(sem_cpp);  // Wait for turn (starts at 1)

        // Write to fifo1
        int fd1 = open("./fifo1", O_WRONLY);  // Blocks until Python opens for reading
        const char *msg = "[C++] Hello Python!";
        write(fd1, msg, strlen(msg) + 1);
        close(fd1);

        sem_post(sem_py);  // Signal Python to read fifo1

        sem_wait(sem_cpp);  // Wait for Python to write to fifo2

        // Read from fifo2
        int fd2 = open("./fifo2", O_RDONLY);  // Blocks until Python opens for writing
        char buffer[1024];
        read(fd2, buffer, sizeof(buffer));
        close(fd2);
        std::cout << "[C++] Received: " << buffer << std::endl;

        sem_post(sem_py);  // Signal Python for next iteration
    }

    // Cleanup
    sem_close(sem_cpp);
    sem_close(sem_py);
    sem_unlink("/sem_cpp_fifo");
    sem_unlink("/sem_py_fifo");
    unlink("./fifo1");
    unlink("./fifo2");
    return 0;
}
