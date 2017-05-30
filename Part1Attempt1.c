#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <X11/Xlib.h>
#include "mandelbrot.h"
#include "mpi.h"

main(int argc, char *argv[] ) {
	double start, finish;
	start=MPI_Wtime(); /*start timer*/

	char message[20];
	int i,rank, size, type=99;
	MPI_Status status;
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

	int totalPixels = imagewidth * imagewidth;
	int pixelsPerProcessor = floor(totalPixels / (size - 1));
	int pixelRemainder = totalPixels % (size - 1);

	if (rank == 0)
	{
		/* open connection with the X server. */
		display = XOpenDisplay(display_name);
		if (display == NULL) 
		{
				fprintf(stderr, "%s: cannot connect to X server '%s'\n",
					argv[0], display_name);
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

		// receives data from the processors
		int remainders[pixelRemainder];
		for (i = 1; i < size; i++) {
			int processedArr[pixelsPerProcessor + 1];
			MPI_Recv(processedArr, pixelsPerProcessor + 1, MPI_INT, MPI_ANY_SOURCE, type, MPI_COMM_WORLD, &status);
			int source = status.MPI_SOURCE;
			
			int startPos = (source - 1) * pixelsPerProcessor;
			int boundaryPos = ( source * pixelsPerProcessor);

			// draws pixels
			for (j = 0; j < pixelsPerProcessor; j++) {
				int yAxis = floor((((source-1) * pixelsPerProcessor) + j) / imagewidth);
				int xAxis = floor((((source-1) * pixelsPerProcessor) + j) % imagewidth);

				if (processedArr[j] == 0) {
	//				printf("%f %f\n",realx,imgx);
					XDrawPoint(display, win, gc, left+xAxis/x, height/2-(imagewidth/2)/x+yAxis/x);
					/*Flush all pending requests to the X server.*/
					XFlush(display);
				}
			}
		}

		// prints last (overhanging) pixels that only some processors did
		for (i = 0; i < pixelRemainder; i++) {
			if (remainders[i] == 0) {
				int yAxis = floor((((size-2) * pixelsPerProcessor) + i) / imagewidth);
				int xAxis = floor((((size-2) * pixelsPerProcessor) + i) % imagewidth);

				XDrawPoint(display, win, gc, left+xAxis/x, height/2-(imagewidth/2)/x+yAxis/x);
			}
		}
		finish=MPI_Wtime(); /*stop timer*/
		printf("%f\n", finish-start);

		/* close the connection to the X server. */
		XCloseDisplay(display);

	} else {
		int startPos = (rank - 1) * pixelsPerProcessor;
		int boundaryPos = ( rank * pixelsPerProcessor);

		int processedArr[pixelsPerProcessor + 1];

		// process pixels
		for (i = startPos; i < boundaryPos; i++) {
			int yAxis = floor(i / imagewidth);
			int xAxis = floor(i % imagewidth);

			realx = s*(xAxis-imagewidth/2) + realcentre;
			imgx = s*(yAxis-imagewidth/2) + imgcentre;
			processedArr[i-startPos] = testmal(realx,imgx,k);
		}

		// if there are overhanging pixels, check if you need to process one then process it
		if (rank <= pixelRemainder) {
			i = (pixelsPerProcessor * size) + rank - 1;

			int yAxis = floor(i / imagewidth);
			int xAxis = floor(i % imagewidth);

			realx = s*(xAxis-imagewidth/2) + realcentre;
			imgx = s*(yAxis-imagewidth/2) + imgcentre;
			processedArr[pixelsPerProcessor] = testmal(realx,imgx,k);
		}


		// send tasks back to master
		MPI_Send(processedArr, sizeof(processedArr)/sizeof(processedArr[0]), MPI_INT, 0, type, MPI_COMM_WORLD);
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
