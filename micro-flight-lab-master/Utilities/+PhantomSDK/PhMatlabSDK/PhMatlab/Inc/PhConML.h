/*****************************************************************************/
//                                                                   
//  Copyright (C) 1992-2011 Vision Research Inc. All Rights Reserved.
//                                                                   
//  The licensed information contained herein is the property of     
//  Vision Research Inc., Wayne, NJ, USA  and is subject to change   
//  without notice.                                                  
//                                                                   
//  No part of this information may be reproduced, modified or       
//  transmitted in any form or by any means, electronic or           
//  mechanical, for any purpose, without the express written         
//  permission of Vision Research Inc.                               
//                                                                   
/*****************************************************************************/


// Control library for the Phantom High Speed Camera
// PhCon.h (public structures, constants and functions)

#ifndef __PHCONINCLUDED__
#define __PHCONINCLUDED__

#ifdef __cplusplus
extern "C"
{
#endif

// Do not define neither _PHCON_ nor _NOPHCON_ in your program to allow
// compiler optimization
#if defined(_PHCON_)
#define EXIMPROC __declspec(dllexport)
#elif !defined(_NOPHCON_)
#define EXIMPROC __declspec(dllimport)
#else
#define EXIMPROC
#endif


#if !defined(_WINDOWS)
// If the platform is not WINDOWS sdk define some common
// types from WIN32 that are used below
typedef unsigned char BYTE, *PBYTE;
typedef unsigned short WORD, *PWORD;
typedef int INT, *PINT, LONG, *PLONG, BOOL, *PBOOL;
typedef unsigned int UINT, *PUINT, DWORD, *PDWORD;
typedef float FLOAT, *PFLOAT;
typedef char *PSTR;
typedef void *PVOID;
#define CALLBACK __stdcall
#define WINAPI __stdcall

//MATLAB ADD START ------------------------------------------------------------------------
typedef int HRESULT;
typedef unsigned int UINT32, *PUINT32;
typedef char CHAR;
#define DECLARE_HANDLE(name) struct name##__ { int unused; }; typedef struct name##__ *name
#define MAX_PATH          260

DECLARE_HANDLE            (HWND);

typedef struct tagPOINT
{
    LONG  x;
    LONG  y;
} POINT, *PPOINT;

typedef struct tagRECT {
    LONG    left;
    LONG    top;
    LONG    right;
    LONG    bottom;
} RECT, *PRECT;
//MATLAB ADD END --------------------------------------------------------------------------

typedef struct tagBITMAPINFOHEADER
{
    DWORD      biSize;
    LONG       biWidth;
    LONG       biHeight;
    WORD       biPlanes;
    WORD       biBitCount;
    DWORD      biCompression;
    DWORD      biSizeImage;
    LONG       biXPelsPerMeter;
    LONG       biYPelsPerMeter;
    DWORD      biClrUsed;
    DWORD      biClrImportant;
} BITMAPINFOHEADER, *PBITMAPINFOHEADER;
typedef struct tagRGBQUAD
{
    BYTE    rgbBlue;
    BYTE    rgbGreen;
    BYTE    rgbRed;
    BYTE    rgbReserved;
} RGBQUAD;
typedef struct tagBITMAPINFO
{
    BITMAPINFOHEADER    bmiHeader;
    //RGBQUAD             bmiColors[1]; (changed to be compatible with MATLAB calllib)
	BYTE             bmiColors[4*256];
} BITMAPINFO, *PBITMAPINFO;
#endif


#if !defined(_TIMEDEFINED_)
#define _TIMEDEFINED_

// A format for small intervals of time: [250 picosecond ... 1 second)
// It is fixed point 0.32 or, with other words, the time in seconds is
// stored multiplied by 4Gig i.e.  4294967296.0 and rounded to int.
typedef UINT32 FRACTIONS, *PFRACTIONS;
/*****************************************************************************/


// The absolute time format used in PC software is TIME64
typedef struct tagTIME64	// A compact format for time 64 bits
							// fixed point (32.32 seconds)
{
    FRACTIONS fractions;	// Fractions of seconds  (resolution 1/4Gig i.e.  cca. 1/4 ns)
							// The fractions of the second are stored here multiplied by 2**32
							// Least significant 2 bits store info about IRIG synchronization
							// bit0 = 0 IRIG synchronized
							// bit0 = 1 not synchronized
							// bit1 = 0 Event input=0 (short to ground)
							// bit1 = 1 Event input=1 (open)
    UINT32 seconds;			// Seconds from Jan 1 1970, compatible with the C library routines
							// (max year: 2038 signed, 2106 unsigned)
							// VS2005 changed the default time_t to 64 bits; here we have to
							// maintain the 32 bits size to remain compatible with the stored
							// file format and the public interfaces
} TIME64, *PTIME64;
/*****************************************************************************/


//Time code according to the standard SMPTE 12M-1999
typedef struct tagTC
{
/*
	Redefined the this part of structure for MATLAB use
    unsigned char framesU:4;        // Units of frames
    unsigned char framesT:2;        // Tens of frames
    unsigned char dropFrameFlag:1;  // Dropframe flag
    unsigned char colorFrameFlag:1; // Colorframe flag
    unsigned char secondsU:4;       // Units of seconds
    unsigned char secondsT:3;       // Tens of seconds
    unsigned char flag1:1;          // Flag 1
    unsigned char minutesU:4;       // Units of minutes
    unsigned char minutesT:3;       // Tens of minutes
    unsigned char flag2:1;          // Flag 2
    unsigned char hoursU:4;         // Units of hours
    unsigned char hoursT:2;         // Tens of hours
    unsigned char flag3:1;          // Flag 3
    unsigned char flag4:1;          // Flag 4
*/
    unsigned char frames;        	
    unsigned char seconds;       	
    unsigned char minutes;       	
    unsigned char hours;         	
    UINT userBitData;   // 32 user bits
}TC, *PTC;
/*****************************************************************************/


// Unpacked representation of SMPTE 12M-1999 Time Code
typedef struct tagTCU
{
	UINT frames;					// Frames
	UINT seconds;					// Seconds
	UINT minutes;					// Minutes
	UINT hours;						// Hours
	BOOL dropFrameFlag;				// Dropframe flag
	BOOL colorFrameFlag;			// Colorframe flag
	BOOL flag1;						// Flag 1
	BOOL flag2;						// Flag 2
	BOOL flag3;						// Flag 3
	BOOL flag4;						// Flag 4
	UINT userBitData;				// 32 user bits
}TCU, *PTCU;
/*****************************************************************************/
#endif


#if !defined(_WBGAIN_)
#define _WBGAIN_
// Color channels adjustment
// intended for the White balance adjustment on color camera
// by changing the gains of the red and blue channels
typedef struct tagWBGAIN
{
    float R;				// White balance, gain correction for red
    float B;				// White balance, gain correction for blue
}
WBGAIN, *PWBGAIN;
/*****************************************************************************/
#endif


// ACQUIPARAMS is the set of control parameters for the camera - upgrade-able
typedef struct tagACQUIPARAMS
{
    UINT ImWidth;			// Image dimensions
    UINT ImHeight;

    UINT FrameRate;			// Frame rate (frames per second)
    UINT Exposure;			// Exposure duration (nanoseconds)
    UINT EDRExposure;		// EDR (extended dynamic range) exposure duration (nanoseconds)
    UINT ImDelay;			// Image delay for the SyncImaging mode (nanoseconds)
							// (not available in all models)
    UINT PTFrames;          // Count of frames to be recorded after the trigger
    UINT ImCount;           // Count of images in this cine (read only field)

    UINT SyncImaging;       // Sync imaging mode: acquire when an external clock rise
							// changed to UINT: possible values SYNC_INTERNAL,
							// SYNC_EXTERNAL, SYNC_LOCKTOIRIG
    UINT AutoExposure;      // Control of the exposure duration from the subject light
							// 0: disabled, 1: enabled, lock at trigger,
							// 3: enabled, active after trigger
    UINT AutoExpLevel;      // Level for autoexposure control
    UINT AutoExpSpeed;      // Speed for autoexposure control
    RECT AutoExpRect;       // Rectangle for autoexposure control

    BOOL Recorded;          // The cine was recorded and is available for playback
							// TriggerTime, RecordedCount, ImCount, FirstIm are
							// valid and final   (read only field)
    TIME64 TriggerTime;     // Trigger time for the recorded cine (read only field)

    UINT RecordedCount;     // initially recorded count; ImCount may be smaller
							// if the cine was partially saved to flash
							// memory (read-only field)
    INT FirstIm;            // First image number of this cine; may be different
							// from PTFrames-ImCount if the cine was partially
							// saved to Non-Volatile memory (read-only field)

    //Frame rate profile
    UINT FRPSteps;          // Supplementary steps in frame rate profile
							// 0 means no frame rate profile
    INT FRPImgNr[16];       // Image number where to change the rate and/or exposure
							// allocated for 16 points (4 available in v7)
    UINT FRPRate[16];       // new value for frame rate (fps)
    UINT FRPExp[16];        // new value for exposure (nanoseconds)
							// The acquisition parameters used before FRPImgNr[0] are the
							// pretrigger parameters from the above fields FrameRate, Exposure

    UINT Decimation;        // Reduce the frame rate when sending the images to i3 through fiber
    UINT BitDepth;          // Bit depth of the cines (read-only field) - usefull for flash; the
							// images from flash have to be requested at the real bit depth.
							// The images from RAM can be requested at different bit depth

    UINT CamGainRed;        // Gains attached to a cine saved in the magazine
    UINT CamGainGreen;      // Normally they tell the White balance at recording time
    UINT CamGainBlue;
    UINT CamGain;           // global gain
    BOOL ShutterOff;        // go to max exposure for piv mode

    UINT CFA;               // Color Filter Array at the recording of the cine
    char CineName[256];     // cine name
    char Description[4096]; // Cine description on max 4095 chars

    UINT FRPShape[16];      // 0: flat,  1 ramp

    // VRI internal note: Size checked structure.
	// Update oldcomp.c if new fields are added  
} ACQUIPARAMS, *PACQUIPARAMS;
/*****************************************************************************/


typedef struct tagIMRANGE
{
    INT First;				//first image number
    UINT Cnt;				//count of images
} IMRANGE, *PIMRANGE;
/*****************************************************************************/


// General settigs of the camera - upgradeable
typedef struct tagCAMERAOPTIONS
{
    // Phantom v4-v6
    // Analog video output
    UINT OSD;               // enable the On Screen Display
    UINT VideoSystem;       // Analog video output system
							// IRIG
    BOOL ModulatedIRIG;     // IRIG input accept modulated signal


    // Fields below are used starting with Phantom v7
    // Trigger
    BOOL RisingEdge;        // TRUE rising, FALSE falling
    UINT FilterTime;        // time constant
    //Memory gate input meaning
    BOOL MemGate;           // TRUE: Memgate, FALSE: pretrigger
    BOOL StartInRec;        // TRUE: Start in recording
							// FALSE:start in preview wait
							//       pretrigger to start the recording
    BOOL Color;             // create color video signal
    BOOL TestImage;         // replace the FBM image by a test image (colored bars)
    //OnScreenDisplay colors
    //RGBQUAD OSDColor[8];  (changed to be compatible with MATLAB calllib)
	BYTE OSDColor[4*8]; 	// set of colors used for the OSD painting
							// Background
							// Wtr mode
							// Trig mode
							// Cst mode
							// Pre-trig mode
							// Id fields
							// Acqui fields
							// Status, cine

    BOOL OSDOpaque;         // OSD test is opaque or transparent
    int OSDLeft;            // limits of the OSD text on screen
    int OSDTop;
    int OSDBottom;
    int ImageX;             // image position on screen
    int ImageY;


    // End of recording automation on Ethernet cameras
    BOOL AutoSaveNVM;       // save the cine to the nvm
    BOOL AutoSaveFile;      // save the cine to a file
    BOOL AutoPlay;          // playback to video
    BOOL AutoCapture;       // restart capture
    CHAR FileName[MAX_PATH];// filename to save (as seen from camera)
    IMRANGE SaveRng;        // image range for the above operations

    BOOL LongReady;         // If FALSE the Ready signal is 1 from the start
							// of recording until the cine is triggered
							// If TRUE the Ready is 1 from the start
							// to the end of recording
    BOOL RealTimeOutput;    // Enable the sending of the acquired image on fiber
    UINT RangeData;         // 0: none, otherwise the size in bytes per image of the range data
							// implemented only 16 bytes

    //external memory slice
    UINT SourceCamSer;      // Source camera serial
    UINT SliceNr;           // slice number (0, 1, 2 ....)
    UINT SliceCnt;          // total count of slices connected to a certain camera
    BOOL FRPi3Trig;         // Frame rate profile start at i3 trigger
    UINT UT;                // OSD: Display time as: 0 LocalTime,  1: Universal Time (GMT)

    int AutoSaveFormat;     // save to flash or direct save to file from camera
							// -8 -16 =  save 8 bit or 16 bits

    UINT SourceCamVer;      // Source camera version (external memory slice)
    UINT RAMBitDepth;       // Set the current bit depth for pixel storage in the camera video RAM
    int VideoTone;          // tone curve to select for video
    int VideoZoom;          // Zoom of the video image: 0: Fit,  1: Zoom1
    int FormatWidth;        // Format rectangle to overlap on the video image
    int FormatHeight;

    UINT AutoPlayCnt;       // repeat count for the playback to video
    UINT OSDDisable;        // selective disable of the analog and digital OSD
    BOOL RecToMag;          // Record to the flash magazine

    BOOL IrigOut;           // Change the Strobe pin to Irig Out at Miro cameras (if TRUE)

	int FormatXOffset;		// x offset for the format rectangle to overlap on the video image
	int FormatYOffset;		// y offset for the format rectangle to overlap on the video image

    //VRI internal note: Size checked structure. Update oldcomp.c if new fields are added    
} CAMERAOPTIONS, *PCAMERAOPTIONS;
/*****************************************************************************/


typedef struct tagCINESTATUS
{
    BOOL Stored;			// Recording of this cine finished
    BOOL Active;			// Acquisition is currently taking place in this cine
    BOOL Triggered;			// The cine has received a trigger
} CINESTATUS, *PCINESTATUS;
/*****************************************************************************/


// PhGetAllIpCameras
typedef struct tagCAMERAID
{
    UINT IP;				// only v4 IP, for v6 to be extended to int64
    UINT Serial;
    char Name[256];
    char Model[256];
} CAMERAID, *PCAMERAID;
/*****************************************************************************/


// Constants area (has to be converted before being imported in MATLAB)
// Include here simple numerical substitutions that will be converted to VAR = value in the MATLAB compatible constants file
// MATLABconstants start


#define PHCONHEADERVERSION 705  // Call PhRegisterClientEx(...,PHCONHEADERVERSION) using this value
								// has to be changed only here; it is the third component of the Phantom files version
/*****************************************************************************/


#define MAXCAMERACNT		63		//maximum count of cameras 
#define MAXCINECNT			64		// maximum count of cine in a multicine camera  
									// including the "preview" cine
#define MAXERRMESS			256		// maximum size of error messages for 
									// PhGetErrorMessage function
#define MAXIPSTRSZ			16		// Maximum IP string length
/*****************************************************************************/


// Predefined cines
#define CINE_DEFAULT		-2		// The number of the default cine
#define CINE_CURRENT		-1		// The cine number used to request live images
#define CINE_PREVIEW		0		// The number of the preview cine
#define CINE_FIRST			1		// The number of the first cine when emulating 
									// a single cine camera
/*****************************************************************************/


// Cines stored in NVM in the Ethernet cameras are directly accesible for
// view (Restore is not necessary). The Flash Cine numbers start at
#define FIRST_FLASH_CINE	101
/*****************************************************************************/


// Selection codes in the functions PhGet(),  PhSet()
// 1. Camera current status information
#define gsHasMechanicalShutter								1025
#define gsHasBlackLevel4									1027
#define gsHasCardFlash										1051
#define gsHas10G											2000
// Continue numbering gsHas... beginning with				2001
/*****************************************************************************/


// 2. Capabilities
#define gsSupportsInternalBlackRef							1026
#define gsSupportsImageTrig									1040
#define gsSupportsCardFlash									1050
#define gsSupportsMagazine									8193
#define gsSupportsHQMode									8194
#define gsSupportsGenlock									8195
#define gsSupportsEDR										8196
#define gsSupportsAutoExposure								8197
#define gsSupportsTurbo										8198
#define gsSupportsBurstMode									8199
#define gsSupportsShutterOff								8200
#define gsSupportsDualSDIOutput								8201
#define gsSupportsRecordingCines							8202
#define gsSupportsV444										8203
#define gsSupportsInterlacedSensor							8204
#define gsSupportsRampFRP									8205
#define gsSupportsOffGainCorrections						8206					
// Continue numbering gsSupports... beginning with			8207
/*****************************************************************************/


// 3. Camera current parameters
#define gsSensorTemperature									1028
#define gsCameraTemperature									1029
#define gsThermoElectricPower								1030
#define gsSensorTemperatureThreshold						1031
#define gsCameraTemperatureThreshold						1032

#define gsVideoPlayCine										1033
#define gsVideoPlaySpeed									1034
#define gsVideoOutputConfig									1035

#define gsMechanicalShutter									1036

#define gsImageTrigThreshold								1041
#define gsImageTrigAreaPercentage							1042
#define gsImageTrigSpeed									1043
#define gsImageTrigMode										1044
#define gsImageTrigRect										1045

#define gsAutoProgress										1046
#define gsAutoBlackRef										1047

#define gsCardFlashSizeK									1052
#define gsCardFlashFreeK									1053
#define gsCardFlashError									1054

#define gsIPAddress											1070
#define gsEthernetAddress									1055
#define gsEthernetMask										1056
#define gsEthernetBroadcast									1057
#define gsEthernetGateway									1058

#define gsLensFocus											1059
#define gsLensAperture										1060
#define gsLensApertureRange									1061
#define gsLensDescription									1062
#define gsLensFocusInProgress								1063
#define gsLensFocusAtLimit									1064

#define gsGenlock											1065
#define gsGenlockStatus										1066

#define gsTurboMode											1068
#define gsModel												1069

#define gsMaxPartitionCount									1071
// Continue numbering gs... beginning with					1072
/*****************************************************************************/


// Selection codes in the functions PhCineGet(), PhCineSet()
#define cgsVideoTone									4097

#define cgsName											4098

#define cgsVideoMarkIn									4099
#define cgsVideoMarkOut									4100

#define cgsIsRecorded                                   4101
#define cgsHqMode                                       4102

#define cgsBurstCount									4103
#define cgsBurstPeriod									4104

#define cgsLensDescription								4105
#define cgsLensAperture									4106
#define cgsLensFocalLength								4107

#define cgsShutterOff									4108

#define cgsTriggerTime									4109

#define cgsTrigTC										4110
#define cgsPbRate										4111
#define cgsTcRate										4112

// Image processing
#define cgsWb											5100
#define cgsBright										5101
#define cgsContrast										5102
#define cgsGamma										5103
#define cgsGammaR										5109
#define cgsGammaB										5110
#define cgsSaturation									5104
#define cgsHue											5105
#define cgsPedestalR									5106
#define cgsPedestalG									5107
#define cgsPedestalB									5108
#define cgsFlare										5111
#define cgsChroma										5112
#define cgsTone											5113
#define cgsUserMatrix									5114
#define cgsEnableMatrices								5115
/*****************************************************************************/


// PhGetVersion constants
#define GV_CAMERA			1
#define GV_FIRMWARE			2
#define GV_FPGA				3
#define GV_PHCON			4
#define GV_CFA				5
#define GV_KERNEL			6
#define GV_MAGAZINE			7
/*****************************************************************************/


// PhGetAuxData selection codes
#define  GAD_TIMEXP         1001    // both image time and exposure
#define  GAD_TIME           1002
#define  GAD_EXPOSURE       1003
#define  GAD_RANGE1         1004    // range data 
#define  GAD_BINSIG         1005    // image binary signals multichannels multisample
									// 8 samples and or channels per byte
#define  GAD_ANASIG         1006    // image analog signals multichannels multisample
									// 2 bytes per sample
#define	 GAD_SMPTETC		1007	// SMPTE time code block (see TC)
#define	 GAD_SMPTETCU		1008	// SMPTE unpacked time code block (see TCU)
/*****************************************************************************/


// PhSetDllsOptions selectors
#define DO_IGNORECAMERAS	1
/*****************************************************************************/


// Set logging options - set the masks for selective dumps from Phantom dlls
// Use them as selection codes for the function PhSetDllsLogOption
#define  SDLO_PHANTOM		100
#define  SDLO_PHCON			101
#define  SDLO_PHINT			102
#define  SDLO_PHFILE		103
#define  SDLO_PHSIG			104
#define  SDLO_PHSIGV		105
#define  SDLO_TORAM			106
/*****************************************************************************/


// Get logging options - get the current settings
// Use them as selection codes for the function PhGetDllsLogOption
#define  GDLO_PHANTOM		200
#define  GDLO_PHCON			201
#define  GDLO_PHINT			202
#define  GDLO_PHFILE		203
#define  GDLO_PHSIG			204
#define  GDLO_PHSIGV		205
#define  GDLO_TORAM			206
/*****************************************************************************/


// Fill the requested data by simulated values
#define  SIMULATED_AUXDATA  0x80000000
/*****************************************************************************/


// Constants for the PhNVMContRec function
// NVM = Flash memory (NonVolatileMemory used for persistent cine store)
#define NVCR_CONT_REC		0x00000001  // Enable continuous recording to NVM mode
#define NVCR_APV			0x00000002  // Enable the automatic playback to video	
#define NVCR_REC_ONCE		0x00000004  // Enable the record once mode
/*****************************************************************************/


// SyncImaging field in ACQUIPARAMS
#define SYNC_INTERNAL		0		// Free-run of camera
#define SYNC_EXTERNAL		1		// Locks to the FSYNC input
#define SYNC_LOCKTOIRIG		2		// Locks to IRIG timecode
/*****************************************************************************/


#if !defined(CFA_VRI)
// Color Filter Array used on the sensor
// In mixed multihead system the gray heads have also some of the msbit set (see XX_GRAY below)
#define CFA_NONE			0		// gray sensor
#define CFA_VRI				1		// gbrg/rggb
#define CFA_VRIV6			2		// bggr/grbg
#define CFA_BAYER			3		// gb/rg
#define CFA_BAYERFLIP		4		// rg/gb
#define CFA_MASK			0xff	// only lsbyte is used for cfa code, the rest is for multiheads
/*****************************************************************************/


// These masks combined with  CFA_VRIV6  describe a mixed (gray&color) image (Phantom v6)
#define TL_GRAY  0x80000000    // Top left head of v6 multihead system is gray
#define TR_GRAY  0x40000000    // Top right head of v6 multihead system is gray
#define BL_GRAY  0x20000000    // Bottom left head of v6 multihead system is gray
#define BR_GRAY  0x10000000    // Bottom right head of v6 multihead system is gray
/*****************************************************************************/


#define ALLHEADS_GRAY 0xF0000000    //(TL_GRAY|TR_GRAY|BL_GRAY|BR_GRAY)
#endif
/*****************************************************************************/


// Analog and digital video output
#define VIDEOSYS_NTSC	        0		// USA analog system
#define VIDEOSYS_PAL	        1		// European analog system

#define VIDEOSYS_720P60         4		// Digital HDTV modes: Progressive
#define VIDEOSYS_720P59DOT9     12
#define VIDEOSYS_720P50	        5
#define VIDEOSYS_1080P30        20
#define VIDEOSYS_1080P29DOT9    28
#define VIDEOSYS_1080P25        21
#define VIDEOSYS_1080P24        36
#define VIDEOSYS_1080P23DOT9    44

#define VIDEOSYS_1080I30        68      // Interlaced
#define VIDEOSYS_1080I29DOT9    76
#define VIDEOSYS_1080I25        69

#define VIDEOSYS_1080PSF30      52      // Progressive split frame
#define VIDEOSYS_1080PSF29DOT9  60
#define VIDEOSYS_1080PSF25      53
#define VIDEOSYS_1080PSF24      84
#define VIDEOSYS_1080PSF23DOT9  92
/*****************************************************************************/


// Notification messages sent to your main window  -  call the notification
// routines to process these messages
#if !defined(NOTIFY_DEVICE_CHANGE)
#define NOTIFY_DEVICE_CHANGE  1325              //(WM_USER+301)   WM_USER=0x0400
#define NOTIFY_BUS_RESET      1326              //(WM_USER+302)
#endif
//MATLABconstants end
/*****************************************************************************/


#if !defined(_NVCRDEFINED_)
#define _NVCRDEFINED_
// Constants for the NVContRec commands and status 
// Write - Read bits 
#define NVCR_CONT_REC   0x00000001  // Enable continuous recording to NVM mode
#define NVCR_APV        0x00000002  // Enable the automatic playback to video
#define NVCR_REC_ONCE   0x00000004  // Enable the record once mode
// Read bits 
#define NVCR_NO_IMAGE   0x00000100  // No image can be sent because of save
									// to NVM during REC_ONCE mode
#endif
/*****************************************************************************/

typedef BOOL (WINAPI *PROGRESSCALLBACK)(UINT CN, UINT Percent);
EXIMPROC HRESULT PhSetDllsOption(UINT optionSelector, PVOID pValue);
EXIMPROC HRESULT PhRegisterClientEx(HWND hWnd, char *pStgPath,
                                    PROGRESSCALLBACK pfnCallback, UINT PhConHeaderVer);
EXIMPROC HRESULT PhUnregisterClient(HWND hWnd);
/*****************************************************************************/


EXIMPROC HRESULT PhGet(UINT CN, UINT Selector, PVOID pVal);
EXIMPROC HRESULT PhSet(UINT CN, UINT Selector, PVOID pVal);
EXIMPROC HRESULT PhCineGet(UINT CN, INT Cine, UINT Selector, PVOID pVal);
EXIMPROC HRESULT PhCineSet(UINT CN, INT Cine, UINT Selector, PVOID pVal);
/*****************************************************************************/


EXIMPROC HRESULT PhGetCineStatus(UINT CN, PCINESTATUS pStatus);
EXIMPROC HRESULT PhGetCineCount(UINT CN, PUINT pRAMCount, PUINT pNVMCount);
EXIMPROC HRESULT PhRecordCine(UINT CN);
EXIMPROC HRESULT PhRecordSpecificCine(UINT CN, int Cine);
EXIMPROC HRESULT PhSendSoftwareTrigger(UINT CN);
EXIMPROC HRESULT PhDeleteCine(UINT CN, INT Cine);
/*****************************************************************************/


EXIMPROC HRESULT PhSetCineParams(UINT CN, INT Cine, PACQUIPARAMS pParams);
EXIMPROC HRESULT PhGetCineParams(UINT CN, INT Cine, PACQUIPARAMS pParams, PBITMAPINFO pBMI);
EXIMPROC HRESULT PhSetSingleCineParams(UINT CN, PACQUIPARAMS pParams);
/*****************************************************************************/


EXIMPROC HRESULT PhGetVersion(UINT CN, UINT VerSel, PUINT pVersion);
EXIMPROC HRESULT PhSetPartitions(UINT CN, UINT Count, PUINT pWeights);
EXIMPROC HRESULT PhGetPartitions(UINT CN, PUINT pCount, PUINT pPartitionSize);
/*****************************************************************************/


EXIMPROC HRESULT PhAddSimulatedCamera(UINT CamVer, UINT Serial);
EXIMPROC HRESULT PhNotifyDeviceChangeCB(PROGRESSCALLBACK pfnCallback);
EXIMPROC HRESULT PhGetIgnoredIp(PUINT pCnt, PUINT pIpAddress);
EXIMPROC HRESULT PhAddIgnoredIp(UINT IpAddress);
EXIMPROC HRESULT PhRemoveIgnoredIp(UINT IpAddress);
EXIMPROC HRESULT PhGetVisibleIp(PUINT pCnt, PUINT pIpAddress);
EXIMPROC HRESULT PhAddVisibleIp(UINT IpAddress);
EXIMPROC HRESULT PhRemoveVisibleIp(UINT IpAddress);
EXIMPROC HRESULT PhMakeAllIpVisible(void);
/*****************************************************************************/


EXIMPROC HRESULT PhGetCameraCount(UINT *pCnt);
EXIMPROC HRESULT PhGetCameraID(UINT CN, UINT *pSerial, char *pCameraName);
EXIMPROC HRESULT PhFindCameraNumber(UINT Serial, UINT *pCN);
/*****************************************************************************/


EXIMPROC HRESULT PhSetCameraName(UINT CN, char *pCameraName);
EXIMPROC HRESULT PhGetCameraTime(UINT CN, PTIME64 pTime64);
EXIMPROC HRESULT PhSetCameraTime(UINT CN, UINT32 Time);
EXIMPROC HRESULT PhGetErrorMessage(int ErrNo, char *pErrText);
EXIMPROC HRESULT PhGetCameraErrMsg(UINT CN, char *pErrText);
EXIMPROC HRESULT PhSetDllsLogOption(UINT SelectCode, UINT Value);
EXIMPROC HRESULT PhGetDllsLogOption(UINT SelectCode, PUINT pValue);
/*****************************************************************************/


EXIMPROC HRESULT PhBlackReference(UINT CN, PROGRESSCALLBACK pfnCallback);
EXIMPROC HRESULT PhBlackReferenceCI(UINT CN, PROGRESSCALLBACK pfnCallback);
EXIMPROC HRESULT PhComputeWB(PBITMAPINFOHEADER pBMIH, PBYTE pPixel, PPOINT pP, int SquareSide,
                             UINT CFA, PWBGAIN pWB, PUINT pSatCnt);
/*****************************************************************************/


EXIMPROC HRESULT PhGetResolutions(UINT CN, PPOINT pRes, PUINT pCnt, PBOOL pCAR, PUINT pADCBits);
EXIMPROC HRESULT PhGetBitDepths(UINT CN, PUINT pCnt, PUINT pBitDepths);
EXIMPROC HRESULT PhGetFrameRateRange(UINT CN, POINT Res, UINT *pMinFrameRate, UINT *pMaxFrameRate);
EXIMPROC HRESULT PhGetExactFrameRate(UINT CamVer, UINT SyncMode, UINT RequestedFrameRate, double *pExactFrameRate);
EXIMPROC HRESULT PhGetExposureRange(UINT CN, UINT FrameRate, UINT *pMinExposure, UINT *pMaxExposure);
/*****************************************************************************/


EXIMPROC HRESULT PhGetCameraOptions(UINT CN, PCAMERAOPTIONS pOptions);
EXIMPROC HRESULT PhSetCameraOptions(UINT CN, PCAMERAOPTIONS pOptions);
/*****************************************************************************/


// Write the calibrations and settings into the camera flash
EXIMPROC HRESULT PhWriteStgFlash(UINT CN, PROGRESSCALLBACK pfnCallback);
/*****************************************************************************/


// Flash (NonVolatileMemory) routines
EXIMPROC HRESULT PhMemorySize(UINT CN, PUINT pDRAMSize, PUINT pNVMSize);
EXIMPROC HRESULT PhNVMErase(UINT CN,  PROGRESSCALLBACK pfnCallback);
EXIMPROC HRESULT PhNVMGetStatus(UINT CN, PUINT pCineCnt, PUINT32 pTime, PUINT pFreeSp);
EXIMPROC HRESULT PhNVMRestore(UINT CN, UINT CineNo,  PROGRESSCALLBACK pfnCallback);
EXIMPROC HRESULT PhNVMContRec(UINT CN, PBOOL pSave, PUINT pOldValue, PUINT pEnable);
EXIMPROC HRESULT PhNVMGetSaveRange(UINT CN, PIMRANGE pRng);
EXIMPROC HRESULT PhNVMSetSaveRange(UINT CN, PIMRANGE pRng);
EXIMPROC HRESULT PhNVMSave(UINT CN, INT Cine, PROGRESSCALLBACK pfnCallback);
EXIMPROC HRESULT PhNVMSaveClip(UINT CN, int Cine, PIMRANGE pRng,
                               UINT Options, PROGRESSCALLBACK pfnCallback);
/*****************************************************************************/


// Video play control
EXIMPROC HRESULT PhVideoPlay(UINT CN, INT CineNr, PIMRANGE pRng, INT PlaySpeed);
EXIMPROC HRESULT PhGetVideoFrNr(UINT CN, PINT pCrtIm);
/*****************************************************************************/


// Routines used from LabView
EXIMPROC HRESULT PhLVRegisterClientEx(char *pStgPath, PROGRESSCALLBACK pfnCallback, UINT PhConHeaderVersion);
EXIMPROC HRESULT PhLVUnregisterClient();
EXIMPROC HRESULT PhLVProgress(UINT CallID, PUINT pPercent, BOOL Continue);
/*****************************************************************************/


EXIMPROC HRESULT PhGetAllIpCameras(PUINT pCnt, PCAMERAID pCam);
EXIMPROC HRESULT PhCheckCameraPool(PBOOL pChanged);
EXIMPROC HRESULT PhParamsChanged(UINT CN, PBOOL pChanged);
/*****************************************************************************/


// PhCon Error Codes
// Informative
#define ERR_Ok                          (0)
#define ERR_SimulatedCamera             (100)
#define ERR_UnknownErrorCode            (101)

#define ERR_BadResolution               (102)
#define ERR_BadFrameRate                (103)
#define ERR_BadPostTriggerFrames        (104)
#define ERR_BadExposure                 (105)
#define ERR_BadEDRExposure              (106)

#define ERR_BufferTooSmall              (107)
#define ERR_CannotSetTime               (108)
#define ERR_SerialNotFound              (109)
#define ERR_CannotOpenStgFile           (110)

#define ERR_UserInterrupt               (111)

#define ERR_NoSimulatedImageFile        (112)
#define ERR_SimulatedImageNot24bpp      (113)

#define ERR_BadParam                    (114)
#define ERR_FlashCalibrationNewer       (115)

#define ERR_ConnectedHeadsChanged       (117)
#define ERR_NoHead                      (118)
#define ERR_NVMNotInstalled             (119)
#define ERR_HeadNotAvailable            (120)
#define ERR_FunctionNotAvailable        (121)
#define ERR_Ph1394dllNotFound           (122)
#define ERR_oldtNotFound                (123)
#define ERR_BadFRPSteps                 (124)
#define ERR_BadFRPImgNr                 (125)
#define ERR_BadAutoExpLevel             (126)
#define ERR_BadAutoExpRect              (127)
#define ERR_BadDecimation               (128)
#define ERR_BadCineParams               (129)
#define ERR_IcmpNotAvailable            (130)
#define ERR_CorrectResetLine			(131)
#define ERR_CSRDoneInCamera             (132)
#define ERR_ParamsChanged               (133)

#define ERR_ParamReadOnly				(134)
#define ERR_ParamWriteOnly				(135)
#define ERR_ParamNotSupported			(136)

#define ERR_IppWarning					(137)
/*****************************************************************************/

// Serious
#define ERR_NULLPointer                 (-200)
#define ERR_MemoryAllocation            (-201)
#define ERR_NoWindow                    (-202)
#define ERR_CannotRegisterClient        (-203)
#define ERR_CannotUnregisterClient      (-204)

#define ERR_AsyncRead                   (-205)
#define ERR_AsyncWrite                  (-206)

#define ERR_IsochCIPHeader              (-207)
#define ERR_IsochDBCContinuity          (-208)
#define ERR_IsochNoHeader               (-209)
#define ERR_IsochAllocateResources      (-210)
#define ERR_IsochAttachBuffers          (-211)
#define ERR_IsochFreeResources          (-212)
#define ERR_IsochGetResult              (-213)

#define ERR_CannotReadTheSerialNumber   (-214)
#define ERR_SerialNumberOutOfRange      (-215)
#define ERR_UnknownCameraVersion        (-216)

#define ERR_GetImageTimeOut             (-217)
#define ERR_ImageNoOutOfRange           (-218)

#define ERR_CannotReadStgHeader         (-220)
#define ERR_ReadStg                     (-221)
#define ERR_StgContents                 (-222)
#define ERR_ReadStgOffsets              (-223)
#define ERR_ReadStgGains                (-224)
#define ERR_NotAStgFile                 (-225)
#define ERR_StgSetupCheckSum            (-226)
#define ERR_StgSetup                    (-227)
#define ERR_StgHardAdjCheckSum          (-228)
#define ERR_StgHardAdj                  (-229)
#define ERR_StgDifferentSerials         (-230)
#define ERR_WriteStg                    (-231)

#define ERR_NoCine                      (-232)
#define ERR_CannotOpenDevice            (-233)
#define ERR_TimeBufferSize              (-234)

#define ERR_CannotWriteCineParams       (-236)

#define ERR_NVMError                    (-250)
#define ERR_NoNVM                       (-251)

#define ERR_FlashEraseTimeout           (-253)
#define ERR_FlashWriteTimeout           (-254)

#define ERR_FlashContents               (-255)
#define ERR_FlashOffsetsCheckSum        (-256)
#define ERR_FlashGainsCheckSum          (-257)

#define ERR_TooManyCameras              (-258)
#define ERR_NoResponseFromCamera        (-259)
#define ERR_MessageFromCamera           (-260)

#define ERR_BadImgResponse              (-261)
#define ERR_AllPixelsBad                (-262)

#define ERR_BadTimeResponse             (-263)
#define ERR_GetTimeTimeOut              (-264)

//Base64 coding and decoding errors
#define ERR_InBlockTooBig               (-270)
#define ERR_OutBufferTooSmall           (-271)
#define ERR_BlockNotValid               (-272)
#define ERR_DataAfterPadding            (-273)
#define ERR_InvalidSlash                (-274)
#define ERR_UnknownChar                 (-275)
#define ERR_MalformedLine               (-276)
#define ERR_EndMarkerNotFound           (-277)

#define ERR_NoTimeData                  (-280)
#define ERR_NoExposureData              (-281)
#define ERR_NoRangeData                 (-282)

#define ERR_NotIncreasingTime           (-283)
#define ERR_BadTriggerTime              (-284)
#define ERR_TimeOut                     (-285)

#define ERR_NullWeightsSum              (-286)
#define ERR_BadCount                    (-287)
#define ERR_CannotChangeRecordedCine    (-288)

#define ERR_BadSliceCount               (-289)
#define ERR_NotAvailable                (-290)
#define ERR_BadImageInterval            (-291)

#define ERR_BadCameraNumber             (-292)
#define ERR_BadCineNumber               (-293)

#define ERR_BadSyncObject               (-294)
#define ERR_IcmpEchoError  			    (-295)

#define ERR_MlairReadFirstPacket		(-296)
#define ERR_MlairReadPacket				(-297)
#define ERR_MlairIncorrectOrder			(-298)

#define ERR_MlairStartRecorder			(-299)
#define ERR_MlairStopRecorder			(-300)
#define ERR_MlairOpenFile				(-301)

#define ERR_CmdMutexTimeOut				(-302)
#define ERR_CmdMutexAbandoned			(-303)

#define ERR_UnsupportedConversion		(-304)

#define ERR_TenGigLostPacket			(-305)

#define ERR_TooManyImgReq				(-306)

#define ERR_BadImRange					(-307)
#define ERR_ImgBufferTooSmall			(-308)
#define ERR_ImgSize0					(-309)

#define ERR_IppError					(-310)
/*****************************************************************************/


///////////////////////////////////////////////////////////////////////////////
//								DEPRECATED									 //
///////////////////////////////////////////////////////////////////////////////


// Set of digital adjustments that can be applied to the image acquired in PhGetImage
typedef struct tagIMPARAMS
{
    //WBGAIN WBGain[4]; (changed to be compatible with MATLAB calllib)
	float WBGainR0;		// Gain adjust on R,B components, for white balance,
	float WBGainB0;		// 1.0 = do nothing,
	float WBGainR1;		// index 0: full image for v4,5,7...TL head for v6
	float WBGainB1;		// index 1, 2, 3 :   TR, BL, BR for v6	
    float WBGainR2;
    float WBGainB2;
    float WBGainR3;
    float WBGainB3;			

    // Analog video out image adjustments;
    // Do not change the digital image transferred to the computer
    int VideoBright;        // [-1024, 1024] neutral 0
    int VideoContrast;      // [1024, 8192] neutral 1024
    float VideoGamma;       // [1024, 4096] neutral 1024
    int VideoSaturation;    // [0, 2048] neutral 1024
    int PedestalR;          // small offset correction, separate on colors
    int PedestalG;
    int PedestalB;

    int VideoHue;           // [-180, 180] neutral 0

    // VRI internal note: Size checked structure.
	// Update oldcomp.c if new fields are added    
} IMPARAMS, *PIMPARAMS;
/*****************************************************************************/


// PhGetImage constants
#define GI_INTERPOLATED 4		// color image, RGB interpolated 
								// extensions for v6 and other multihead systems
#define GI_ONEHEAD      8		// multihead system, bit=1: select image from one head 
#define GI_HEADMASK  0xf0		// head number: 0:TL 1:TR 2:BL 3:BR.. (up to 16 heads, 
								// v6 has 4 heads)   -  bits 4...7 of the Options
								// extensions for v7
//#define GI_BPP12				0x100   // BitsPerPixel = 12 for the requested image - not implemented, not used
#define GI_BPP16	0x200		// BitsPerPixel = 16 for the requested image 

#define GI_PACKED	0x100		// return the pixels as read from camera; possible packed (10 bits: 4 pixels in 5 bytes) 

#define GI_DENORMALIZED	0x2000	// Leave the pixels as read from camera, possible with black level != 0 and white level != Maximum  
								//Default is to scale the pixels value so the black level is at 0 and white level at Maximum Pixel Value
/*****************************************************************************/


// PhGetI3Info
#define  GII_SOURCECAMERA   1		// Source camera number (camera connected through 
//fiber to memory slices)
#define  GII_SLICECOUNT     2		// Slice Count 
#define  GII_SLICELIST      3		// List of camera numbers to access the slices
#define  GII_VIRTUALCN      4		// Camera number of the virtual device
/*****************************************************************************/


EXIMPROC HRESULT PhRegisterClient(HWND hWnd, char *pStgPath);
EXIMPROC HRESULT PhRegisterClientCB(HWND hWnd, char *pStgPath,
                                    PROGRESSCALLBACK pfnCallback);
/*****************************************************************************/


EXIMPROC HRESULT PhSetImageParameters(UINT CN, PIMPARAMS pParams);
EXIMPROC HRESULT PhGetImageParameters(UINT CN, PIMPARAMS pParams);
EXIMPROC HRESULT PhSetCineImageParameters(UINT CN, int Cine, PIMPARAMS pIPPar);
EXIMPROC HRESULT PhGetCineImageParameters(UINT CN, int Cine, PIMPARAMS pIPPar);
/*****************************************************************************/


EXIMPROC HRESULT PhMeasureWB(UINT CN, PPOINT pP, int SquareSide, UINT ImageCount,
                             UINT Options, PROGRESSCALLBACK pfnCallback, PWBGAIN pWB, PUINT pSatCnt);
EXIMPROC HRESULT PhGetAuxData(UINT CN, UINT Sel, INT Cine, PIMRANGE pRng, PBYTE pData);
EXIMPROC HRESULT PhGetCameraModel(UINT CamVer, PSTR pModel);
/*****************************************************************************/


EXIMPROC HRESULT PhGetImage(UINT CN, PINT pCine, PIMRANGE pRng, UINT Options, PBYTE pPixel);
/*****************************************************************************/

EXIMPROC HRESULT PhRestoreCameraStatus(UINT CN);
EXIMPROC HRESULT PhNotifyBusReset(void);
EXIMPROC HRESULT PhBusReset(void);
EXIMPROC HRESULT PhCheckNotification(void);
EXIMPROC HRESULT PhStreamToVideo(UINT CN);
EXIMPROC HRESULT PhStreamTo1394(UINT CN);
/*****************************************************************************/


EXIMPROC HRESULT PhGetI3Info(UINT CN, UINT Selection, PVOID pData);
/*****************************************************************************/


EXIMPROC HRESULT PhSetSimulatedCamera(UINT CamVer, UINT CFA);
EXIMPROC HRESULT PhNotifyDeviceChange(void);
EXIMPROC HRESULT PhGetVersions(UINT CN, UINT *pCameraVersion, UINT *pFirmwareVersion, UINT *pDllVersion, UINT *pCFA);
EXIMPROC HRESULT PhBlackBalance(UINT CN, PROGRESSCALLBACK pfnCallback);//synonym for PhBlackReference, not to be used
EXIMPROC HRESULT PhGetCameraResolutions(UINT CN, PPOINT pRes, PUINT pCnt);
EXIMPROC HRESULT PhSetAcquisitionParameters(UINT CN, PACQUIPARAMS pParams);
EXIMPROC HRESULT PhGetAcquisitionParameters(UINT CN, PACQUIPARAMS pParams, PBITMAPINFO pBMI);
EXIMPROC HRESULT PhGetCineParameters(UINT CN, PACQUIPARAMS pParams, PBITMAPINFO pBMI);
EXIMPROC HRESULT PhHasCine(UINT CN, PBOOL pHasCine);    //has at least one cine

EXIMPROC HRESULT PhGetPreviewImage(UINT CN, PBITMAPINFOHEADER pBMIH, UINT Options, PBYTE pPixel);
EXIMPROC HRESULT PhGetRecordedImage(UINT CN, PBITMAPINFOHEADER pBMIH, INT ImageNo, UINT Options, PBYTE pPixel);
EXIMPROC HRESULT PhGetRecordedBlock(UINT CN, PBITMAPINFOHEADER pBMIH, INT ImageNo, UINT Options, PBYTE pPixel);

EXIMPROC HRESULT PhGetCineTime(UINT CN, PTIME64 pTime, PFRACTIONS pExposure);
EXIMPROC HRESULT PhGetTriggerTime(UINT CN, PTIME64 pTriggerTime);

EXIMPROC HRESULT PhSearchForAllCameras(PROGRESSCALLBACK pfnCallback);	//deprecated, to be replaced by PhGetAllIpCameras

// Function for starting/stopping ITT recorder app
EXIMPROC HRESULT PhStartIttRecorder();
EXIMPROC HRESULT PhStopIttRecorder();
/*****************************************************************************/

#undef EXIMPROC     // avoid conflicts with other similar headers

#ifdef __cplusplus
}
#endif
#endif
