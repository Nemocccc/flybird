#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winsock2.h>
#include <windows.h>  // For CreateThread
#include <time.h>

#pragma comment(lib, "ws2_32.lib")  // Link with ws2_32.lib

#define PORT 12345
#define BUFFER_SIZE 1024
#define GAP 9 // Wall gap
#define UPPER_BOUNDARY 1
#define LOWER_BOUNDARY 26
#define DELAY 200 // Delay in milliseconds

typedef struct {
    int x;
    int y;
} Coord;

typedef struct {
    Coord pos;
    int score;
} Bird;

typedef struct {
    Coord walls[3];
    Bird bird;
} GameState;

void init_game(GameState *state) {
    state->bird.pos.x = 22;
    state->bird.pos.y = 10;
    state->bird.score = 0;

    state->walls[0] = (Coord){40, 10};
    state->walls[1] = (Coord){60, 6};
    state->walls[2] = (Coord){80, 8};
}

int update_game(GameState *state, int action) {
    // Update bird position
    if (action == 1) {
        state->bird.pos.y -= 1;
    } else {
        state->bird.pos.y += 1;
    }

    // Update wall positions
    for (int i = 0; i < 3; i++) {
        state->walls[i].x--;
        if (state->walls[i].x < 10) {
            state->walls[i].x = 80;
            state->walls[i].y = rand() % 13 + 5;
            state->bird.score++;
        }
    }

    return 0;  // Return 0 to indicate success
}

int check_collision(GameState *state) {
    Bird *bird = &state->bird;
    Coord *walls = state->walls;

    for (int i = 0; i < 3; i++) {
        if (bird->pos.x >= walls[i].x && bird->pos.x < walls[i].x + 5) {
            if (bird->pos.y <= walls[i].y || bird->pos.y >= walls[i].y + GAP) {
                return 1; // Collision detected
            }
        }
    }

    if (bird->pos.y < UPPER_BOUNDARY || bird->pos.y > LOWER_BOUNDARY) {
        return 1; // Bird out of bounds
    }

    return 0; // No collision
}

DWORD WINAPI handle_client(LPVOID client_socket) {
    SOCKET new_socket = *(SOCKET*)client_socket;
    char buffer[BUFFER_SIZE] = {0};
    GameState game_state;

    init_game(&game_state);

    while (1) {
        int valread = recv(new_socket, buffer, BUFFER_SIZE, 0);
        if (valread <= 0) break;

        if (strcmp(buffer, "reset") == 0) {
            init_game(&game_state);
        } else if (strcmp(buffer, "step") == 0) {
            update_game(&game_state, 0);  // No jump
        } else if (strcmp(buffer, "jump") == 0) {
            update_game(&game_state, 1);  // Jump
        }

        // Check for collisions
        int game_over = check_collision(&game_state);

        char response[BUFFER_SIZE];
        snprintf(response, BUFFER_SIZE,
            "{\"game_over\": %d, \"bird\": {\"x\": %d, \"y\": %d, \"score\": %d}, "
            "\"walls\": [{\"x\": %d, \"y\": %d}, {\"x\": %d, \"y\": %d}, {\"x\": %d, \"y\": %d}]}",
            game_over,
            game_state.bird.pos.x, game_state.bird.pos.y, game_state.bird.score,
            game_state.walls[0].x, game_state.walls[0].y,
            game_state.walls[1].x, game_state.walls[1].y,
            game_state.walls[2].x, game_state.walls[2].y);

        send(new_socket, response, strlen(response), 0);

        // Delay to control game speed
        Sleep(DELAY);

        memset(buffer, 0, BUFFER_SIZE);
    }

    closesocket(new_socket);
    return 0;
}

int main() {
    WSADATA wsaData;
    SOCKET server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    srand(time(NULL));

    // Initialize Winsock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        printf("WSAStartup failed\n");
        return 1;
    }

    // Create socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        printf("Socket creation error\n");
        WSACleanup();
        return 1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind the socket to the network address and port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) == SOCKET_ERROR) {
        printf("Bind failed\n");
        closesocket(server_fd);
        WSACleanup();
        return 1;
    }

    // Start listening for connections
    if (listen(server_fd, 3) == SOCKET_ERROR) {
        printf("Listen failed\n");
        closesocket(server_fd);
        WSACleanup();
        return 1;
    }

    printf("Server is running on port %d\n", PORT);

    while (1) {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen)) == INVALID_SOCKET) {
            printf("Accept failed\n");
            closesocket(server_fd);
            WSACleanup();
            return 1;
        }

        // Create a new thread for each client
        DWORD thread_id;
        HANDLE thread_handle = CreateThread(NULL, 0, handle_client, &new_socket, 0, &thread_id);
        if (thread_handle == NULL) {
            printf("Error creating thread\n");
            closesocket(new_socket);
        } else {
            CloseHandle(thread_handle);  // We don't need to keep the handle after creating the thread
        }
    }

    closesocket(server_fd);
    WSACleanup();

    return 0;
}