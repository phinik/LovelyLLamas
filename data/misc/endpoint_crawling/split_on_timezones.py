import pandas as pd
import os
from datetime import datetime
import pytz

def get_current_utc_offset(timezone_str: str) -> float:
    """
    Get the current UTC offset for a timezone in hours.
    Returns offset as float (e.g., 2.0 for UTC+2, -4.5 for UTC-4:30)
    """
    try:
        tz = pytz.timezone(timezone_str)
        # Get current offset in hours
        offset = tz.utcoffset(datetime.now()) / pd.Timedelta(hours=1)
        return offset
    except Exception as e:
        print(f"Error getting offset for {timezone_str}: {e}")
        return None

def split_csv_by_timezone(input_file: str, output_dir: str = "timezone_splits") -> None:
    """
    Split a CSV file into multiple files based on timezone UTC offset.
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Directory to store the split files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Add UTC offset column
    df['UTC_Offset'] = df['Timezone'].apply(get_current_utc_offset)
    
    # Get unique UTC offsets
    offsets = df['UTC_Offset'].dropna().unique()
    
    # Create summary data
    summary = []
    
    # Split data by UTC offset and save to separate files
    for offset in offsets:
        # Filter data for this offset
        offset_data = df[df['UTC_Offset'] == offset].copy()
        
        # Create filename - replace only the decimal in the offset number, not the extension
        if offset >= 0:
            offset_str = str(abs(offset)).replace('.', '_')
            filename = f"UTC_plus_{offset_str}.csv"
        else:
            offset_str = str(abs(offset)).replace('.', '_')
            filename = f"UTC_minus_{offset_str}.csv"
            
        output_path = os.path.join(output_dir, filename)
        
        # Add timezone names to summary
        timezone_names = offset_data['Timezone'].unique()
        
        # Save to CSV (excluding the UTC_Offset column)
        offset_data.drop('UTC_Offset', axis=1).to_csv(output_path, index=False)
        
        # Add to summary
        summary.append({
            'UTC_Offset': f"UTC{'+' if offset >= 0 else ''}{offset}",
            'Cities': len(offset_data),
            'Filename': filename,
            'Included_Timezones': ', '.join(sorted(timezone_names))
        })
        
        print(f"Created {filename} with {len(offset_data)} cities")
    
    # Create summary file
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Cities', ascending=False)
    summary_df.to_csv(os.path.join(output_dir, "_summary.csv"), index=False)
    print(f"\nCreated summary file with {len(summary_df)} unique UTC offsets")
    
    # Print summary
    print("\nSummary of files created:")
    print(summary_df[['UTC_Offset', 'Cities', 'Filename']].to_string())
    
    # Print detailed timezone information
    print("\nDetailed timezone groupings:")
    for _, row in summary_df.iterrows():
        print(f"\n{row['UTC_Offset']} ({row['Cities']} cities):")
        print(f"Timezones: {row['Included_Timezones']}")

if __name__ == "__main__":
    split_csv_by_timezone(f"{os.getcwd()}/data/city_data_tz.csv")