#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <X11/Xlib.h>
#include "mandelbrot.h"
#include "mpi.h"

main(int argc, char *argv[] ) {

	char message[20];
	int i,rank, size, type=99;
	MPI_Status status;
	
	int provided;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int x,j,imagewidth,left,outside,r,k;
	float s,realcentre,imgcentre,realx,imgx;

	Display* display;             /* pointer to X Display structure.           */
	int screen_num;               /* number of screen to place the window on.  */
	Window win;                   /* pointer to the newly created window.      */
	unsigned int display_width, display_height;  /* height and width of the X display. */
	unsigned int width, height;   /* height and width for the new window.      */
	char *display_name = getenv("DISPLAY");  /* address of the X display.      */
	GC gc;                        /* GC (graphics context) used for drawing    */
								/*  in our window.                           */

	k = -10;
	imagewidth = 1000;
	x = 1;

	if (argc != 4) {
		fprintf(stderr,"Usage: %s boundary_level image_width scaling_factor\n",argv[0]);
	} else {
		k = atoi(argv[1]);
		imagewidth = atoi(argv[2]);
		x = atoi(argv[3]);
	}

	r = 2;
	s = (float)(2*r)/imagewidth;
	realcentre = 0.0;
	imgcentre = 0.0;
	left = 0;

	if (rank == 0)
	{

		/* open connection with the X server. */
		display = XOpenDisplay(display_name);
		if (display == NULL) 
		{
			fprintf(stderr, "%s: cannot connect to X server '%s'\n", argv[0], display_name);
			exit(1);
		}

		/* get the geometry of the default screen for our display. */
		screen_num = DefaultScreen(display);
		display_width = DisplayWidth(display, screen_num);
		display_height = DisplayHeight(display, screen_num);

		/* make the new window occupy much of the screen's size. */
		width = (display_width * 0.75);
		height = (display_height * 0.75);
	//  	printf("window width - '%d'; height - '%d'\n", width, height);

		/* create a simple window, as a direct child of the screen's   */
		/* root window. Use the screen's white color as the background */
		/* color of the window. Place the new window's top-left corner */
		/* at the given 'x,y' coordinates.                             */
		win = create_simple_window(display, width, height, 0, 0);

		/* allocate a new GC (graphics context) for drawing in the window. */
		gc = create_gc(display, win, 0);
		XSync(display, False);

		// create allocations

		double start, finish;
		start=MPI_Wtime(); /*start timer*/
		int currenti;

		// ---------------------------------------------------------------------------
		// 
		// MARK: Split the image into sections so that every processor gets 
		// imagewidth/n processes and so that processor with rank (starting at 1) would 
		// get process rank, rank + n, rank + 2n, ... rank + (i * n).
		// 
		// ---------------------------------------------------------------------------

		// allocations must have size rows since MPI_Scatter scatters to all processors
		// including processor 0
		int allocations[size][imagewidth+1];
		
		int currentPerson = 1;
		int currentIndex = 0;

		int finalPerson = size-1;
		for (currenti = 0; currenti < imagewidth; currenti++) {
			allocations[currentPerson][currentIndex] = currenti;

			if (currentPerson == finalPerson) {
				currentPerson = 1;
				currentIndex += 1;
			} else {
				currentPerson += 1;
			}
		}

		// ---------------------------------------------------------------------------
		// 
		// MARK: Set last value of each subarray to be the actual length of the array
		// (so that we don't read values that haven't been set)
		// 
		// ---------------------------------------------------------------------------

		for (i = 1; i < size; i++) {
			if (i < currentPerson) {
				allocations[i][imagewidth] = currentIndex + 1;
			} else {
				allocations[i][imagewidth] = currentIndex;
			}
		}

		// ---------------------------------------------------------------------------
		
		MPI_Scatter(allocations, imagewidth + 1, MPI_INT, allocations[0], imagewidth + 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		// ---------------------------------------------------------------------------
		// 
		// MARK: Recieve items to be drawn
		// 
		// ---------------------------------------------------------------------------

		int calculatedArr[imagewidth*imagewidth];
		for (i = 1; i < size; i++) {

			MPI_Recv(calculatedArr, imagewidth * imagewidth, MPI_INT, MPI_ANY_SOURCE, type, MPI_COMM_WORLD, &status);
			int source = status.MPI_SOURCE;
			int allocationSize = allocations[source][imagewidth];

			for(j=0;j<allocationSize;j++) {
				int reali = allocations[source][j];

				for(k=0;k<imagewidth;k++) {
					if (calculatedArr[(j*imagewidth)+k] == 0) {
						XDrawPoint(display, win, gc, left+k/x, height/2-(imagewidth/2)/x+reali/x);
						/*Flush all pending requests to the X server.*/
						XFlush(display);
					}
				}
			}
		}

		// ---------------------------------------------------------------------------

		finish=MPI_Wtime(); /*start timer*/
		printf("%f\n", finish-start);
		// sleep(300);

		// ---------------------------------------------------------------------------

	} else {

		// ---------------------------------------------------------------------------
		
		double startProcessor=MPI_Wtime(); /*start timer*/
		int allocation[imagewidth+1];

		MPI_Scatter(NULL, imagewidth + 1, MPI_INT, allocation, imagewidth + 1, MPI_INT, 0, MPI_COMM_WORLD);

		int allocationSize = allocation[imagewidth];

		// ---------------------------------------------------------------------------
		// 
		// MARK: Calculate whether pixels belong to the mandelbrot set
		// 
		// ---------------------------------------------------------------------------
		
		int calculatedArr[imagewidth*imagewidth]; 
		for(i=0;i<allocationSize;i++) {
			for(j=0;j<imagewidth;j++) {
				realx = s*(j-imagewidth/2) + realcentre;
				imgx = s*(allocation[i]-imagewidth/2) + imgcentre;
				calculatedArr[(i*imagewidth)+j] = testmal(realx,imgx,k);
			}
		}

		double finishProcessor=MPI_Wtime(); /*start timer*/
		// printf("rank %d finished in %f seconds \n", rank, finishProcessor-startProcessor);

		// ---------------------------------------------------------------------------

		MPI_Send(calculatedArr, imagewidth*imagewidth, MPI_INT, 0, type, MPI_COMM_WORLD);
	
		// ---------------------------------------------------------------------------
	}
	MPI_Finalize();
}		 

int testmal(float realx, float imgx, int k)
{
	int i;
	float re,im,re2,im2;
	
	re = realx;
	im = imgx;

	for(i=0;i<2-k;i++)
	{
		re2 = re*re;
		im2 = im*im;
		if ((re2+im2) > 256)
		{
			return 1;
		}
		im = 2*re*im + imgx;
		re = re2 - im2 + realx;
	}
	return(0);
}			

Window create_simple_window(Display* display, int width, int height, int x, int y)
{
  	int screen_num = DefaultScreen(display);
  	int win_border_width = 2;
  	Window win;

  	/* create a simple window, as a direct child of the screen's */
  	/* root window. Use the screen's black and white colors as   */
  	/* the foreground and background colors of the window,       */
  	/* respectively. Place the new window's top-left corner at   */
  	/* the given 'x,y' coordinates.                              */
  	win = XCreateSimpleWindow(display, RootWindow(display, screen_num),
                            x, y, width, height, win_border_width,
                            BlackPixel(display, screen_num),
                            WhitePixel(display, screen_num));

  	/* make the window actually appear on the screen. */
  	XMapWindow(display, win);

  	/* flush all pending requests to the X server. */
  	XFlush(display);

  	return win;
}

GC create_gc(Display* display, Window win, int reverse_video)
{ 
  	GC gc;                                /* handle of newly created GC.  */
  	unsigned long valuemask = 0;          /* which values in 'values' to  */
  	                                      /* check when creating the GC.  */
  	XGCValues values;                     /* initial values for the GC.   */
  	unsigned int line_width = 2;          /* line width for the GC.       */
  	int line_style = LineSolid;           /* style for lines drawing and  */
  	int cap_style = CapButt;              /* style of the line's edje and */
  	int join_style = JoinBevel;           /*  joined lines.               */
  	int screen_num = DefaultScreen(display);
  
  	gc = XCreateGC(display, win, valuemask, &values);
  	if (gc < 0) 
	{
        	fprintf(stderr, "XCreateGC: \n");
  	}

  	/* allocate foreground and background colors for this GC. */
  	if (reverse_video) 
	{
    		XSetForeground(display, gc, WhitePixel(display, screen_num));
    		XSetBackground(display, gc, BlackPixel(display, screen_num));
  	}
  	else 
	{
    		XSetForeground(display, gc, BlackPixel(display, screen_num));
    		XSetBackground(display, gc, WhitePixel(display, screen_num));
  	}

  	/* define the style of lines that will be drawn using this GC. */
  	XSetLineAttributes(display, gc,
                     line_width, line_style, cap_style, join_style);

  	/* define the fill style for the GC. to be 'solid filling'. */
  	XSetFillStyle(display, gc, FillSolid);

  	return gc;
}
