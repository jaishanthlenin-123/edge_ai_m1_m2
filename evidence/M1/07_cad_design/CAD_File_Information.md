# CAD File Information: ENCLOSURE_JAISHANTH.f3d

**Component:** Raspberry Pi Camera Enclosure (IDRP Hardware Component)
**Designer:** Jaishanth
**Location in Repository:**
- Primary Evidence Path: [`evidence/M1/07_cad_design/ENCLOSURE_JAISHANTH.f3d`](./ENCLOSURE_JAISHANTH.f3d)
- Root Path: [`ENCLOSURE_JAISHANTH.f3d`](../../../ENCLOSURE_JAISHANTH.f3d)

---

## 1. File Summary & Identification

| Property | Value |
| :--- | :--- |
| **Filename** | `ENCLOSURE_JAISHANTH.f3d` |
| **File Extension** | `.f3d` |
| **CAD System** | Autodesk Fusion 360 |
| **Format Specification** | Autodesk Fusion Document Archive (`hybrid-design` standard) |
| **Geometric Kernel** | Autodesk Neutron / Autodesk Shape Manager (ASM 232.5.0.65535 NT) |
| **Internal Archive Format** | Open Packaging ZIP Container (RFC 1951 / RFC 8878 Zstandard) |
| **Compression Algorithm** | Zstandard (Method 93) |
| **Exact File Size** | **874,983 bytes** (854.48 KB / ~875 KB) |
| **Creation / Save Timestamp** | `Sun Nov 08 18:06:12 2026` |
| **Integrity Status** | **VERIFIED – Valid, Accessible & Non-Corrupted** |

---

## 2. Cryptographic Checksums

The following hashes verify that the CAD archive has not been modified, truncated, or corrupted:

```
========================================================================================
                              CRYPTOGRAPHIC CHECKSUMS
========================================================================================
Algorithm   Checksum
----------------------------------------------------------------------------------------
SHA-256     84444cee13753f74e86be3e41a7b0b865c8727c2576d06237835fc374914a4e1
MD5         5f00dc2a2e8801f53b70868a8e7b5c1a
========================================================================================
```

---

## 3. Internal Architecture & Data Streams

Inspection of `ENCLOSURE_JAISHANTH.f3d` reveals 68 internal entries organized into functional subsystems:

```
ENCLOSURE_JAISHANTH.f3d (ZIP Package)
├── Manifest.dat                                       # Document root manifest
├── Properties.dat                                     # Document type definition
└── FusionAssetName[Active]/
    ├── Manifest.dat                                   # Active asset metadata
    ├── Previews/
    │   ├── big.png                                    # Genuine 1024x1024 3D isometric render
    │   └── small.png                                  # 256x256 thumbnail render
    ├── FusionDesignSegmentType1/
    │   ├── BulkStream.dat                             # Parametric timeline & body definitions
    │   └── MetaStream.dat                             # Component hierarchy & feature metadata
    ├── FusionBrowserSegmentType1/
    │   ├── BulkStream.dat                             # Browser tree structures
    │   └── MetaStream.dat                             # Browser node names
    ├── Breps.BlobParts/                               # Exact ACIS Boundary Representation (B-rep)
    │   ├── BREP.e8c862e3-7c07-46a5-8343-*.smb         # ASM solid body 1 (Lid)
    │   ├── BREP.d37f2298-554c-42ab-8258-*.smb         # ASM solid body 2 (Base Housing)
    │   └── BREP.4613a4ce-2a3b-4d26-a626-*.smb         # ASM solid body 3 (Camera PCB Assembly)
    ├── ParaMeshGeometry.BlobParts/                    # 32 UDPMESH tessellation blocks
    │   └── ParaMeshGeometry.*.paramesh                # Faceted representations for graphics
    ├── OGS.BlobFolder/OGS/DefaultScene/
    │   ├── world                                      # One Graphics System scene graph
    │   └── stream_mesh_000                            # Viewport vertex buffer (44,555 vertices)
    └── ProteinAssets.BlobParts/
        └── ProteinAsset.e677a04d-*.protein            # Physical material & appearance definitions
```

---

## 4. How to Open and View the CAD File

### Option 1: Autodesk Fusion 360 (Native Application)
1. Launch **Autodesk Fusion 360** (desktop client).
2. Click **File** $\rightarrow$ **Open...**
3. Select **Open from my computer** and browse to `ENCLOSURE_JAISHANTH.f3d`.
4. Alternatively, click **Upload** in the Data Panel and drag-and-drop `ENCLOSURE_JAISHANTH.f3d`.

### Option 2: Autodesk Viewer (Free Browser-Based, No Software Required)
1. Open a web browser and navigate to [https://viewer.autodesk.com/](https://viewer.autodesk.com/).
2. Sign in with a free Autodesk account.
3. Drag and drop `ENCLOSURE_JAISHANTH.f3d` into the upload zone.
4. Rotate, pan, zoom, measure dimensions, and explode the model directly in the browser.

### Option 3: Exporting to STEP / STL for 3D Printing
In Fusion 360:
- **STEP:** Click **File** $\rightarrow$ **Export...** $\rightarrow$ select **STEP Files (*.step, *.stp)**.
- **STL for 3D Slicing:** Right-click the component in the browser tree $\rightarrow$ select **Save As Mesh** $\rightarrow$ choose **STL (Binary)** $\rightarrow$ **OK**.

---

## 5. Preservation Policy Compliance

- **No Format Conversion:** The original `.f3d` binary archive has been preserved without conversion to lossy formats.
- **No Overwriting:** The original root file is preserved, and an identical verified copy is located in `evidence/M1/07_cad_design/`.
- **Integrity Verified:** Hashes and file sizes match across all storage locations.

