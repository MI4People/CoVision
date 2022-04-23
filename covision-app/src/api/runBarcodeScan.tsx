import Quagga from 'quagga';

export type BarcodeScanResult = {
  codeResult: {
    code?: string;
  };
};
export const runBarcodeScan = (src: string): Promise<BarcodeScanResult> => {
  return new Promise((resolve) =>
    Quagga.decodeSingle(
      {
        src,
        locator: {
          halfSample: true,
          patchSize: 'large', // x-small, small, medium, large, x-large
          debug: {
            showCanvas: true,
            showPatches: false,
            showFoundPatches: false,
            showSkeleton: false,
            showLabels: false,
            showPatchLabels: false,
            showRemainingPatchLabels: false,
            boxFromPatches: {
              showTransformed: true,
              showTransformedBox: true,
              showBB: true,
            },
          },
        },
        numOfWorkers: 4,
        decoder: {
          readers: ['code_128_reader', 'ean_reader'],
          debug: {
            drawBoundingBox: true,
            showFrequency: true,
            drawScanline: true,
            showPattern: true,
          },
        },
        locate: true,
      },
      resolve
    )
  );
};
