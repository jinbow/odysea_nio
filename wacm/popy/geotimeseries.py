

def regrid(dd,target_time):
    """
    dd is a pandas Series with time index
    target_time is a pd.date_range, the data will be regridded onto this time
    
    use Kriging method
    
    return:
    
    pandas.Series with regridded data
    
    """
    
    