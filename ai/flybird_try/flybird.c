#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<conio.h>
#include<windows.h>

#define GAP 9//柱子上下间隔
#define DIS 20//柱子左右间隔
#define BOOL int//无他，布尔类型用起来更直观些
#define TRUE 1
#define FALSE 0
#define upper_boundry 1//上边界
#define lower_boundry 26//下边界


typedef struct bird
{   
    COORD pos;//COORD是表示坐标的结构体.
    int score;
}BIRD;

void CheckWall(COORD wall[], BIRD * bird); //显示柱子
void PrtBird(BIRD * bird); //显示小鸟
int CheckWin(COORD * wall, BIRD * bird);//检测小鸟是否碰墙或者超出上下边界。
void Begin(BIRD * bird);//显示上下边界和分数
BOOL SetConsoleColor(unsigned int wAttributes);//设置颜色
void Gotoxy(int x, int y);//定位光标
void HideCursor();//隐藏光标，减少闪屏

//思路：小鸟只走上下，柱子往左边走
int main()
{
    BIRD bird = {{22, 10}, 0};//小鸟的初始位置
    COORD wall[3] = {{40, 10}, {60, 6}, {80, 8}};//柱子的初始位置和高度
    int i;
    char user_hit;

    while (CheckWin(wall, &bird))//游戏循环
    {       
        Begin(&bird);//刷新
        CheckWall(wall, &bird);//显示柱子
        PrtBird(&bird);//显示小鸟
        Sleep(200);//古希腊掌管帧率的神

        if (kbhit())
        {
            user_hit = getch();
            if (user_hit == ' ' ||  user_hit== 'w' || user_hit == 'W')//用户输入空格
            {
                bird.pos.Y -= 1;//小鸟向上一格
            }
        }
        else
        {
            bird.pos.Y += 1;//小鸟向下一格
        }
        for(i = 0; i < 3; ++i)
        {
            wall[i].X--;//柱子向左一格
        }
    }

    return 0;
}
  
//函数功能：显示/刷新柱子
//参数：柱子位置的列表
void CheckWall(COORD wall[], BIRD * bird)
{
    int i;
    HideCursor();
    srand(time(NULL));//用时间作为随机数种子
    COORD temp = {wall[2].X + DIS, rand() % 13 + 5};//随机产生一个新的柱子

    if (wall[0].X < 10)//超出左边界
    {
        wall[0] = wall[1];//最左侧柱子消失，第二个柱子变成第一个
        wall[1] = wall[2];//第三个->第二个
        wall[2] = temp;//新产生的变成第三个
        (bird -> score)++;//写在这里，每次柱子刷新，分数加一（成绩记录的是越过的柱子数量） 
    }  

    for (i = 0; i < 3; ++i)
    {
        //显示上半部分柱子墙
        temp.X = wall[i].X+1;//向右缩进一格显示图案，因为判断小鸟是否撞墙的时候是判断小鸟的横坐标是否大于等于柱子横坐标
        SetConsoleColor(0x0C);//设置黑色背景
        for (temp.Y = 2; temp.Y < wall[i].Y; temp.Y++)//从第二行开始显示
        {
            Gotoxy(temp.X, temp.Y);
            printf("#####");
        }
        temp.X--;
        Gotoxy(temp.X, temp.Y);
        printf(" #####");
        //显示下半部分柱子墙
        temp.Y += GAP;
        Gotoxy(temp.X, temp.Y);
        printf(" #####"); 
        temp.X++;//向右缩进一格
        temp.Y++;//在下一行显示下面的图案
        for (; (temp.Y) < lower_boundry; temp.Y++)//一直显示到25行
        {
            Gotoxy(temp.X, temp.Y);  
            printf("#####");
        }
    } 
}
  
//函数功能：显示小鸟
void PrtBird (BIRD * bird)
{
    SetConsoleColor(0x0E);//设置黑色背景,亮黄色前景
    Gotoxy(bird->pos.X, bird->pos.Y);
    printf("o->");
}

//函数功能:检测小鸟是否碰到墙体或者超出上下边界,是则返回0,否则分数加1并返回1
int CheckWin (COORD * wall, BIRD * bird)
{
    if (bird->pos.X >= wall->X && bird -> pos.X < wall -> X + 5)//小鸟的横坐标进入柱子坐标范围
    {
        if (bird->pos.Y <= wall->Y || bird->pos.Y >= wall->Y + GAP)
        {
            return 0;//小鸟的纵坐标碰到上下柱子,则返回0
        }
    }
    if (bird->pos.Y < upper_boundry || bird->pos.Y > lower_boundry)
    {
        return 0;//小鸟的位置超出上下边界,则返回0
    }
    //(bird -> score)++;//写在这里，每刷新一次分数加一 

    return 1;      
}

//函数功能:显示上下边界和分数 
void Begin (BIRD * bird)
{
    system("cls");
    Gotoxy(0, lower_boundry);//第二十六行显示下边界
    printf("==========================================================================");
    Gotoxy(0, upper_boundry);
    printf("==========================================================================");
    SetConsoleColor(0x0E);//设置黑色背景，亮黄色前景
    printf("\n%4d", bird -> score);//第一行显示分数
}

//函数功能:定位光标 v  
void Gotoxy (int x , int y)
{
    COORD pos = {x, y};
    HANDLE hOutput = GetStdHandle(STD_OUTPUT_HANDLE);//获得标准输出设备句柄
    SetConsoleCursorPosition(hOutput, pos);
}

//函数功能:设置颜色
//一共有16种文字颜色,16种背景颜色,组合有256种。传入的参数值应当小于256
//字节的低4位控制前景色,高4位控制背景色,高亮+红+绿+蓝
BOOL SetConsoleColor (unsigned int wAttributes)
{
    HANDLE houtput = GetStdHandle (STD_OUTPUT_HANDLE);
    if (houtput == INVALID_HANDLE_VALUE)
    {
        return FALSE ;
    }
    return SetConsoleTextAttribute (houtput, wAttributes);     
}

//两数功能:隐藏光标,避免闪屏现象,提高游戏体验
void HideCursor()
{
    HANDLE handle = GetStdHandle (STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO CursorInfo ;
    GetConsoleCursorInfo (handle, &CursorInfo);//获取控制台光标信息
    CursorInfo .bVisible = 0 ;//隐藏控制台光标
    SetConsoleCursorInfo (handle, &CursorInfo);//设置控制台光标状态
}