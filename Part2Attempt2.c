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
	int i, index, rank, size, type=99;
	MPI_Status status;
	
	int provided;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int x,j,imagewidth,left,outside,r,k,lowMemoryMode, stepSize, initialStepSize;
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
	stepSize = 10;
	lowMemoryMode = 0;

	int numOfSlaves = size-1;


	if (argc < 4) {
		fprintf(stderr,"Usage: %s boundary_level image_width scaling_factor, step_size, initial_step_size '-d'\n",argv[0]);
	} else {
		// set up GDB
		if (argc == 5){
			stepSize = atoi(argv[4]);
		}
		if (argc == 6){
			initialStepSize = atoi(argv[5]);
		}
		if (argc == 7) {
			if (strcmp(argv[argc-1], "-d") == 0) {
				setupDebugging(rank);
			}
		}

		k = atoi(argv[1]);
		imagewidth = atoi(argv[2]);
		x = atoi(argv[3]);

		
	}

	initialStepSize = (imagewidth/2)/numOfSlaves;

	if (stepSize > initialStepSize) {
		fprintf(stderr,"stepSize cannot be greater than the initialStepSize\n",argv[0]);
		exit(1);
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
		win = create_simple_window(display, imagewidth, imagewidth, 0, 0);

		/* allocate a new GC (graphics context) for drawing in the window. */
		gc = create_gc(display, win, 0);
		XSync(display, False);

		// XResizeWindow(display, win, imagewidth, imagewidth);


		// create allocations

		double startTime=MPI_Wtime(); //start timer
		int currenti;

		// ---------------------------------------------------------------------------
		// 
		// MARK: Split the image into sections so that every processor gets 
		// imagewidth/n processes and so that processor with rank (starting at 1) would 
		// get process rank, rank + n, rank + 2n, ... rank + (i * n).
		// 
		// ---------------------------------------------------------------------------

		// allocations must have size num_of_rows since MPI_Scatter scatters to all processors
		// including processor 0

		int (*allocations)[initialStepSize] = malloc(size * initialStepSize * sizeof(int));
		// int allocations = (int (*)) p;
		
		int currentPerson = 1;
		int currentIndex = 0;

		int finalPerson = size-1;

		int initialAllocationSize = initialStepSize * numOfSlaves;
		for (currenti = 0; currenti < initialAllocationSize; currenti++) {
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
		// Scatter allocations
		// 
		// ----------------------------------------------------------------------------
		MPI_Scatter(allocations, initialStepSize, MPI_INT, allocations[0], initialStepSize, MPI_INT, 0, MPI_COMM_WORLD);
		
		free(allocations);
		allocations = NULL;

		// ---------------------------------------------------------------------------
		// 
		// Setup state and variables for dynamic task delegation
		// 
		// ---------------------------------------------------------------------------

		int state[numOfSlaves];
		memset( state, 0, numOfSlaves * sizeof(int) );

		int calculatedArr[initialStepSize][imagewidth];

		int nextAllocationStartPos = initialAllocationSize;

		int numOfSequentials = 0;

		int returnBuf[numOfSlaves][stepSize];

		MPI_Request reqs[numOfSlaves];

		int leftOverRows = imagewidth - (initialAllocationSize);

		 // the number processes that will still be sent out (ceiled since we will need to send out a half process if there is an overhang with the number of processors we have ) 
		int expectedReceives = numOfSlaves + ceil(leftOverRows / stepSize);

		int processorFinishedFlag = -1;

		// Receive and draw chunks
		for (i = 0; i < expectedReceives; i++) {

			// Recv task from processor
			MPI_Recv(calculatedArr, initialStepSize*imagewidth, MPI_INT, MPI_ANY_SOURCE, type, MPI_COMM_WORLD, &status);
			int source = status.MPI_SOURCE;

			if (i < expectedReceives - numOfSlaves) {

				// if there are tasks left allocate tasks and send them
				allocateTask(returnBuf[source-1], nextAllocationStartPos, stepSize, numOfSlaves);
				MPI_Isend(returnBuf[source-1], stepSize, MPI_INT, source, type, MPI_COMM_WORLD, &reqs[source-1]);
			} else {
				// otherwise send (I've finished flag)
				MPI_Isend(&processorFinishedFlag, stepSize, MPI_INT, source, type, MPI_COMM_WORLD, &reqs[source-1]);
			}

			int lastStart = state[source-1]; 
			int allocationSize;

			// if the last allocation was the initial allocation use different step size and determine the start position from the rank
			if (lastStart == 0) {
				lastStart = source - 1;
				allocationSize = initialStepSize;
			} else {
				allocationSize = stepSize;
			}

			// draws the points received
			for(j=0;j<allocationSize;j++) {
				int reali = lastStart + (j * numOfSlaves);

				for(k=0;k<imagewidth;k++) {
					if (calculatedArr[j][k] == 0) {
						XDrawPoint(display, win, gc, left+k/x, height/2-(imagewidth/2)/x+reali/x);
						XFlush(display);
					}
				}
			}

			// sets the state at which the processor it
			state[source-1] = nextAllocationStartPos;

			// Sets variables to track what the next task will be
			if (numOfSequentials < numOfSlaves - 1) {
				nextAllocationStartPos += 1;
				numOfSequentials++;
			} else {
				nextAllocationStartPos += 1 + numOfSlaves * (stepSize - 1);
				numOfSequentials = 0;
			}

		}
		printf("%f\n", MPI_Wtime() - startTime);
	} else {
		
		double startTime=MPI_Wtime(); /*start timer*/
		double commTime = 0.0;
		int allocation[initialStepSize];

		int allocationSize = initialStepSize;
		int calculatedArr[initialStepSize][imagewidth]; 

		// receives scattered tasks
		MPI_Scatter(NULL, initialStepSize, MPI_INT, allocation, initialStepSize, MPI_INT, 0, MPI_COMM_WORLD);

		// calculates points if there are points left
		while (allocation[0] != -1) {
			for(i=0;i<allocationSize;i++) {
				for(j=0;j<imagewidth;j++) {
					realx = s*(j-imagewidth/2) + realcentre;
					imgx = s*(allocation[i]-imagewidth/2) + imgcentre;
					calculatedArr[i][j] = testmal(realx,imgx,k);
				}
			}

			// Sends processed tasks and receives new tasks
			allocationSize = stepSize;
			double startCommTime=MPI_Wtime(); /*start timer*/
			MPI_Send(calculatedArr, initialStepSize*imagewidth, MPI_INT, 0, type, MPI_COMM_WORLD);
			MPI_Recv(allocation, stepSize*imagewidth, MPI_INT, 0, type, MPI_COMM_WORLD, &status);
			commTime += (MPI_Wtime() - startCommTime); /*start timer*/
		}
		printf("rank %d Comm Time = %f\n", rank, commTime);
		printf("rank %d Time Taken = %f\n", rank, MPI_Wtime() - startTime);
	}
	MPI_Finalize();
}		 

void allocateTask(int *buf, int start, int length, int stride)
{
	int i;
	int index = 0;
	for (i = start; index < length; i+=stride, index++) {
		buf[index] = i;
	}
}			

void setupDebugging(int rank)
{
	// Function to setup debug environment. puts master in endless while loop and sets
	// up VSCode launch.js file properly with the correct attachment PID

	char hostname[256];
	gethostname(hostname, sizeof(hostname));
	int pid = getpid();
	printf("rank %d - PID %d on %s ready for attach\n", rank, pid, hostname);
	fflush(stdout);
	if (rank == 0) {

		// MODIFY launch.json to use rank0's attachment PID
		int triedCreating = 0;

		FILE *filePtr = fopen("./.vscode/launch.json", "w");

		if (!filePtr)
		{
			// try to create the file
			fprintf(stderr, "Error: ./.vscode/launch.json was not found");
			exit(EXIT_FAILURE);
		}

		// scan for the float
		char jsonFront[300] = "\
		{ \
			\"version\": \"0.2.0\", \
			\"configurations\": [ \
				{ \
					\"type\": \"gdb\", \
					\"request\": \"attach\", \
					\"name\": \"GDB Attach to PID\", \
					\"executable\": \"./bin/executable\", \
					\"target\": \"";
		
		char jsonBack[300] = "\", \
					\"cwd\": \"${workspaceRoot}\", \
					\"ssh\": { \
						\"host\": \"HOST_IP\", \
						\"cwd\": \"/home/YOUR_STUDENT_NUMBER/\", \
						\"password\": \"YOUR_PASSWORD\", \
						\"user\": \"YOUR_STUDENT_NUMBER\" \
					} \
				} \
			] \
		}";

		char pidStr[20]; 

		char combinedJson[650];
		sprintf(pidStr, "%d", pid);
		strcpy(combinedJson, jsonFront);
    	strcat(combinedJson, pidStr);
    	strcat(combinedJson, jsonBack);

		fwrite(combinedJson, strlen(combinedJson), 1, filePtr);

		fclose(filePtr);

		// endless loop p0 ready for gdb intercept
		int i = 0;
		while (0 == i)
			sleep(2);

	}
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
