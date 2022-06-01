Instructions for use:

To use the Face detection program you need to set path in matlab to the bin directory of this zip file
"FaceDetect.dll" is used by versions earlier than 7.1 while "FaceDetect.mexw32" is used by later versions
The two files "cv100.dll" and "cxcore.dll" should be placed in the same directory as the other files.


Requirements for compiling:

Matlab 7.0.0 R14 or Matlab 7.5.0 R2007b
Microsoft visual studio 2003 or 2005

Tested on:
Matlab 7.0.0 R14 with Microsoft Visual C/C++ version 7.1 in C:\Program Files\Microsoft Visual Studio .NET 2003
Matlab 7.5.0 R2007b with Microsoft Visual C++ 2005 in C:\Program Files\Microsoft Visual Studio 8

Instructions for compiling:
1. Setup Mex compiler:
   Type "mex -setup" in the command window of matlab. Follow the instructions and choose the appropriate compiler.
   The native C compiler with Matlab did not compile this program. MS visual studio compilers are preferred.
2. Change path to the /src/ directory and issue the command 
   mex FaceDetect.cpp -I../Include/ ../lib/*.lib -outdir ../bin/
   The compiled files are stored in the bin directory. Place these output files along with "cv100.dll" and "cxcore.dll"
   in desired directory and set path appropriately in matlab.

NOTE: compiling with Visual Studio 2005 version 8 compilier requires that a compiler sepcific dll be included along with 
the zip file. All the compiling on this zip are with visual studio 2003 version 7.1 compiler.