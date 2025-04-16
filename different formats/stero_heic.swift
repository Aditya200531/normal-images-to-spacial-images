import Foundation
import AppKit
import UniformTypeIdentifiers

func combineImages(leftImg: CGImage, rightImg: CGImage, outputPath: String) {
    let newImageURL = URL(fileURLWithPath: outputPath)
    // Create a HEIC destination that can store 2 images
    guard let destination = CGImageDestinationCreateWithURL(newImageURL as CFURL, UTType.heic.identifier as CFString, 2, nil) else {
        print("Failed to create image destination")
        return
    }
    
    let imageWidth = CGFloat(leftImg.width)
    let imageHeight = CGFloat(leftImg.height)
    let fovHorizontalDegrees: CGFloat = 55
    let fovHorizontalRadians = fovHorizontalDegrees * (.pi / 180)
    let focalLengthPixels = 0.5 * imageWidth / tan(0.5 * fovHorizontalRadians)
    let baseline = 65.0 // in millimeters
    
    // Build camera intrinsics metadata array
    let cameraIntrinsics: [CGFloat] = [
        focalLengthPixels, 0, imageWidth / 2,
        0, focalLengthPixels, imageHeight / 2,
        0, 0, 1
    ]
    
    let properties: [CFString: Any] = [
        kCGImagePropertyExifDictionary: [
            // Any EXIF properties you need
        ],
        kCGImagePropertyHEIFDictionary: [
            // Embedding the camera intrinsics in a vendor-specific key
            "CameraModel": ["Intrinsics": cameraIntrinsics]
        ],
        // Stereo pair metadata
        kCGImagePropertyTIFFDictionary: [
            "StereoPair": [
                "LeftImageIndex": 0,
                "RightImageIndex": 1
            ]
        ]
    ]
    
    // Add left image
    CGImageDestinationAddImage(destination, leftImg, properties as CFDictionary)
    // Add right image
    CGImageDestinationAddImage(destination, rightImg, properties as CFDictionary)
    
    // Finalize the destination
    if CGImageDestinationFinalize(destination) {
        print("Combined HEIC image saved at \(outputPath)")
    } else {
        print("Failed to finalize HEIC destination")
    }
}

// Usage example (you’d load your images as CGImage):
if let leftImage = NSImage(contentsOfFile: "/path/to/left_view.jpg")?.cgImage(forProposedRect: nil, context: nil, hints: nil),
   let rightImage = NSImage(contentsOfFile: "/path/to/right_view.jpg")?.cgImage(forProposedRect: nil, context: nil, hints: nil) {
    combineImages(leftImg: leftImage, rightImg: rightImage, outputPath: "/path/to/combined_output.heic")
}