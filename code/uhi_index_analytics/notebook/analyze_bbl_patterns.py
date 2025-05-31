import pandas as pd
import re
import numpy as np
from collections import Counter

def analyze_bbl_patterns(dataframe, column_name):
    """
    Analyze all patterns in the BBL column and provide a detailed report
    
    Args:
        dataframe: Pandas DataFrame containing the BBL column
        column_name: Name of the column containing BBL values
    
    Returns:
        DataFrame with pattern analysis
    """
    # Create a copy to avoid modifying the original
    df = dataframe.copy()
    
    # Convert all values to string and handle NaN/None
    df[column_name] = df[column_name].fillna('NA').astype(str)
    
    # Define pattern extraction function
    def extract_pattern(value):
        # Replace actual digits with 'D' to identify the pattern
        digit_pattern = re.sub(r'\d', 'D', value)
        
        # Identify special characters and their positions
        special_chars = re.findall(r'[^a-zA-Z0-9\s]', value)
        char_count = Counter(special_chars)
        
        # Check for specific patterns
        contains_semicolon = ';' in value
        contains_comma = ',' in value
        contains_hyphen = '-' in value
        contains_slash = '/' in value
        contains_space = ' ' in value
        contains_question = '?' in value
        
        # Create pattern descriptor
        pattern_desc = f"Pattern: {digit_pattern}"
        if special_chars:
            pattern_desc += f" | Special chars: {dict(char_count)}"
        
        # Check for length consistency if multiple values
        if contains_semicolon or contains_comma:
            delimiter = ';' if contains_semicolon else ','
            parts = value.split(delimiter)
            lengths = [len(part.strip()) for part in parts]
            pattern_desc += f" | Multiple values with lengths: {lengths}"
        
        return pattern_desc
    
    # Apply pattern extraction
    df['pattern'] = df[column_name].apply(extract_pattern)
    
    # Group by pattern and count occurrences
    pattern_counts = df.groupby('pattern').size().reset_index(name='count')
    
    # Add example values for each pattern
    pattern_examples = df.groupby('pattern')[column_name].apply(
        lambda x: list(x.sample(min(3, len(x))).values)
    ).reset_index(name='examples')
    
    # Merge counts and examples
    pattern_analysis = pd.merge(pattern_counts, pattern_examples, on='pattern')
    
    # Sort by count in descending order
    pattern_analysis = pattern_analysis.sort_values('count', ascending=False)
    
    return pattern_analysis

def suggest_regex_patterns(pattern_analysis):
    """
    Suggest regex patterns based on the pattern analysis
    
    Args:
        pattern_analysis: DataFrame with pattern analysis
        
    Returns:
        DataFrame with suggested regex patterns
    """
    suggestions = []
    
    for _, row in pattern_analysis.iterrows():
        pattern = row['pattern']
        examples = row['examples']
        count = row['count']
        
        # Extract pattern characteristics
        contains_semicolon = ';' in pattern
        contains_comma = ',' in pattern
        contains_hyphen = '-' in pattern
        contains_slash = '/' in pattern
        contains_space = ' ' in pattern
        
        # Suggest regex pattern
        suggested_regex = ""
        
        if 'Multiple values' in pattern:
            if contains_semicolon:
                suggested_regex = r'[\d\-\/\s]+(?:;[\d\-\/\s]+)+'
            elif contains_comma:
                suggested_regex = r'[\d\-\/\s]+(?:,[\d\-\/\s]+)+'
        elif contains_hyphen and contains_slash:
            suggested_regex = r'\d+[\-\/]\d+[\-\/]\d+'
        elif contains_hyphen:
            suggested_regex = r'\d+\-\d+\-\d+'
        elif contains_slash:
            suggested_regex = r'\d+\/\d+\/\d+'
        elif contains_space:
            suggested_regex = r'\d+\s+\d+\s+\d+'
        else:
            # Basic pattern for just digits
            suggested_regex = r'\d+'
            
        suggestions.append({
            'pattern': pattern,
            'count': count,
            'examples': examples,
            'suggested_regex': suggested_regex
        })
    
    return pd.DataFrame(suggestions)

# Example usage:
# df = pd.read_csv('../energy.csv')
# patterns = analyze_bbl_patterns(df, 'NYC Borough, Block and Lot (BBL)')
# regex_suggestions = suggest_regex_patterns(patterns)
# print(regex_suggestions)
