import numpy as np

def find_peaks(x):
        """
        Finds the maximum value in each channel and returns the location and value.
        
        Args:
            x: rank-4 tensor [batch, height, width, channels]

        Returns:
            peaks: rank-3 tensor [batch, 3, channels] -> (x, y, val) for each channel
        """
        b, h, w, c = x.shape  # unpack shape

        # Flatten height and width into one dimension: [batch, H*W, channels]
        flattened = x.reshape(b, h * w, c)

        # Find indices of maxima along spatial dimension
        idx = np.argmax(flattened, axis=1)  # [batch, channels]

        # Convert flat index to (row, col)
        rows = idx // w
        cols = idx % w

        # Max values per channel
        vals = np.max(flattened, axis=1)  # [batch, channels]

        # Stack results into shape [batch, 3, channels]
        pred = np.stack([cols.astype(float), rows.astype(float), vals], axis=1)

        return pred