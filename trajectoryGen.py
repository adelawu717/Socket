import matlab.engine
import os

def main():
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Change to the directory containing your MATLAB script
    eng.cd('/Users/adelawu/Desktop/OnTrack_Rehab/ArmTroi-main/data')  # Replace with the actual path

    try:
        # Run the MATLAB script
        eng.new(nargout=0)
        eng.cd('/Users/adelawu/Desktop/OnTrack_Rehab/ArmTroi-main/src/pointCloudGeneration')
        eng.pointCloudGenerate_overall(nargout=0)
        eng.cd('/Users/adelawu/Desktop/OnTrack_Rehab/ArmTroi-main/src/stateEstimation')
        eng.stateEstimation_overall(nargout=0)
        eng.quit()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop MATLAB engine
        eng.quit()


if __name__ == "__main__":
    main()