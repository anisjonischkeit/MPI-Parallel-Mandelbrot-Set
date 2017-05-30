/* Function Prototypes */

int testmal(float,float,int);
Window create_simple_window(Display *,int,int,int,int);
GC create_gc(Display *,Window,int);
void drawMandelbrot(int argc, char *argv[], int size, int type);

void setupDebugging(int rank);
void allocateTask(int *buf, int start, int length, int stride);
