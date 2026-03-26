import 'dart:async';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await dotenv.load(fileName: ".env");
  _cameras = await availableCameras();
  runApp(const FaceVerificationApp());
}

class FaceVerificationApp extends StatelessWidget {
  const FaceVerificationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Verification',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(scaffoldBackgroundColor: Colors.black),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  bool _isInitialized = false;
  bool _isProcessingFrame = false;
  bool _permissionDenied = false;

  // Verification state
  bool _isVerifying = false;
  String? _verificationMessage;
  Color _verificationColor = Colors.white;

  final String _backendUrl =
      'http://${dotenv.env['BACKEND_IP']}:8080/verify';

  // ML Kit Face Detector
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: true,
      enableLandmarks: true,
      performanceMode: FaceDetectorMode.fast,
    ),
  );

  List<Face> _detectedFaces = [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed && _permissionDenied) {
      _checkPermissionAndInit();
    }
  }

  Future<void> _checkPermissionAndInit() async {
    final status = await Permission.camera.status;
    if (status.isGranted) {
      setState(() => _permissionDenied = false);
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      setState(() => _permissionDenied = true);
      return;
    }

    if (_cameras.isEmpty) return;

    final frontCamera = _cameras.firstWhere(
      (cam) => cam.lensDirection == CameraLensDirection.front,
      orElse: () => _cameras.first,
    );

    _cameraController = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.nv21
          : ImageFormatGroup.bgra8888,
    );

    try {
      await _cameraController!.initialize();
      if (!mounted) return;

      await _cameraController!.startImageStream(_processImage);

      setState(() => _isInitialized = true);
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  Future<void> _processImage(CameraImage cameraImage) async {
    if (_isProcessingFrame || _isVerifying) return;
    _isProcessingFrame = true;

    try {
      final inputImage = _convertCameraImage(cameraImage);
      if (inputImage == null) {
        _isProcessingFrame = false;
        return;
      }

      final faces = await _faceDetector.processImage(inputImage);

      if (mounted) {
        setState(() => _detectedFaces = faces);
      }
    } catch (e) {
      debugPrint('Error processing image: $e');
    }

    _isProcessingFrame = false;
  }

  InputImage? _convertCameraImage(CameraImage image) {
    final camera = _cameraController!.description;

    final rotation =
        InputImageRotationValue.fromRawValue(camera.sensorOrientation);
    if (rotation == null) return null;
    if (image.planes.isEmpty) return null;

    final size = Size(image.width.toDouble(), image.height.toDouble());

    if (Platform.isAndroid) {
      if (image.planes.length < 3) return null;
      final nv21 = _yuv420ToNv21(image);
      return InputImage.fromBytes(
        bytes: nv21,
        metadata: InputImageMetadata(
          size: size,
          rotation: rotation,
          format: InputImageFormat.nv21,
          bytesPerRow: image.width,
        ),
      );
    }

    // iOS — bgra8888
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null || format != InputImageFormat.bgra8888) return null;

    final allBytes = WriteBuffer();
    for (final Plane plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }

    return InputImage.fromBytes(
      bytes: allBytes.done().buffer.asUint8List(),
      metadata: InputImageMetadata(
        size: size,
        rotation: rotation,
        format: format,
        bytesPerRow: image.planes.first.bytesPerRow,
      ),
    );
  }

  Uint8List _yuv420ToNv21(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final int halfW = width ~/ 2;
    final int halfH = height ~/ 2;

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final int uvPixelStride = uPlane.bytesPerPixel ?? 1;
    final int uvRowStride = uPlane.bytesPerRow;

    final nv21 = Uint8List(width * height + halfW * halfH * 2);
    int pos = 0;

    // Y plane — strip row‑stride padding
    for (int row = 0; row < height; row++) {
      final int rowStart = row * yPlane.bytesPerRow;
      for (int col = 0; col < width; col++) {
        nv21[pos++] = yPlane.bytes[rowStart + col];
      }
    }

    // VU interleaved (NV21 order)
    for (int row = 0; row < halfH; row++) {
      final int rowStart = row * uvRowStride;
      for (int col = 0; col < halfW; col++) {
        final int idx = rowStart + col * uvPixelStride;
        nv21[pos++] = vPlane.bytes[idx];
        nv21[pos++] = uPlane.bytes[idx];
      }
    }

    return nv21;
  }

  Future<void> _verifyFace() async {
    if (_isVerifying) return;

    setState(() {
      _isVerifying = true;
      _verificationMessage = 'Capturing & Verifying...';
      _verificationColor = Colors.blue;
    });

    try {
      // Temporarily stop stream to take a high-res photo for the backend
      await _cameraController!.stopImageStream();

      final file = await _cameraController!.takePicture();
      final bytes = await file.readAsBytes();

      // Send to Go Backend
      var request = http.MultipartRequest('POST', Uri.parse(_backendUrl));
      request.files.add(
        http.MultipartFile.fromBytes('image', bytes, filename: 'live.jpg'),
      );

      final streamedResponse = await request.send().timeout(
        const Duration(seconds: 15),
      );
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final bool verified = data['verified'] ?? false;
        final String msg = data['message'] ?? 'Unknown error';

        setState(() {
          _verificationMessage = verified ? 'VERIFIED: $msg' : 'REJECTED: $msg';
          _verificationColor = verified ? Colors.green : Colors.red;
        });
      } else {
        setState(() {
          _verificationMessage = 'Server Error: ${response.statusCode}';
          _verificationColor = Colors.orange;
        });
      }
    } catch (e) {
      debugPrint('Verification error: $e');
      setState(() {
        _verificationMessage = 'Network Error. Check IP and server.';
        _verificationColor = Colors.orange;
      });
    } finally {
      // Ensure we start the image stream back up
      if (mounted) {
        setState(() => _isVerifying = false);
        try {
          await _cameraController!.startImageStream(_processImage);
        } catch (e) {
          debugPrint('Error restarting image stream: $e');
        }
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    if (_isInitialized) {
      // Must ignore async errors if we're disposing
      _cameraController?.stopImageStream().ignore();
    }
    _cameraController?.dispose();
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(body: _buildBody());
  }

  Widget _buildBody() {
    if (_permissionDenied) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.camera_alt_outlined,
                size: 64,
                color: Colors.white54,
              ),
              const SizedBox(height: 16),
              const Text(
                'Camera permission is required',
                style: TextStyle(fontSize: 18, color: Colors.white),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () => openAppSettings(),
                child: const Text('Open Settings'),
              ),
            ],
          ),
        ),
      );
    }

    if (!_isInitialized || _cameraController == null) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        // Camera preview without stretching
        Positioned.fill(
          child: Container(
            color: Colors.black,
            child: Center(
              child: CameraPreview(_cameraController!),
            ),
          ),
        ),

        // Overlay blur when verifying
        if (_isVerifying)
          Container(
            color: Colors.black54,
            child: const Center(
              child: CircularProgressIndicator(color: Colors.blue),
            ),
          ),

        // Face count overlay (top)
        Positioned(
          top: MediaQuery.of(context).padding.top + 16,
          left: 0,
          right: 0,
          child: Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              decoration: BoxDecoration(
                color: _detectedFaces.isNotEmpty
                    ? Colors.green.withValues(alpha: 0.8)
                    : Colors.red.withValues(alpha: 0.6),
                borderRadius: BorderRadius.circular(24),
              ),
              child: Text(
                _detectedFaces.isEmpty
                    ? 'No face detected'
                    : '${_detectedFaces.length} face${_detectedFaces.length > 1 ? 's' : ''} detected',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ),

        // Verification Status Message
        if (_verificationMessage != null)
          Positioned(
            top: MediaQuery.of(context).padding.top + 70,
            left: 16,
            right: 16,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _verificationColor.withValues(alpha: 0.85),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                _verificationMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),

        // Action panel + Verify Button (bottom)
        Positioned(
          bottom: MediaQuery.of(context).padding.bottom + 24,
          left: 16,
          right: 16,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Show ML Kit face info if detected
              if (_detectedFaces.isNotEmpty && !_isVerifying)
                Container(
                  padding: const EdgeInsets.all(16),
                  margin: const EdgeInsets.only(bottom: 16),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.7),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: _detectedFaces.asMap().entries.map((entry) {
                      final face = entry.value;
                      return Text(
                        'Face ${entry.key + 1}: '
                        'Smile ${(face.smilingProbability ?? 0) > 0.5 ? "😊" : "😐"} '
                        '| Eyes ${(face.leftEyeOpenProbability ?? 0) > 0.5 && (face.rightEyeOpenProbability ?? 0) > 0.5 ? "👁" : "😑"}',
                        style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 13,
                        ),
                      );
                    }).toList(),
                  ),
                ),

              // Verify Button
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: (_detectedFaces.length == 1 && !_isVerifying)
                      ? _verifyFace
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    disabledBackgroundColor: Colors.grey.withValues(alpha: 0.5),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(28),
                    ),
                  ),
                  child: const Text(
                    'Capture & Verify Face',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
