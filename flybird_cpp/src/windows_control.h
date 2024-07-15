#ifndef WINDOWS_CONTROL_H
#define WINDOWS_CONTROL_H

class windows_control{
    public:
        void HideCursor();
        void Gotoxy(int x, int y);
        int SetConsoleColor(unsigned int wAttributes);
};

#endif