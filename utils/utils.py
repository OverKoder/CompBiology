from pandas import DataFrame

def get_by_system(data: DataFrame, system: str or list):
    """
    Returns all the abundance samples of a given system or systems (A or B)

    Args:
        data (DataFrame): The abundance dataframe
        system (str or list): The system to group by, can be 1 value or many

    Raises:
        TypeError: data is not a DataFrame
        TypeError: system is not a list or a string

    Returns:
        (DataFrame) the data grouped by system
    """

    if not isinstance(data, DataFrame): raise TypeError("data is not a DataFrame")
    if not (isinstance(system, list) or isinstance(system, str)): raise TypeError("system is not a list or string")

    # If it is a list
    if isinstance (system, list):
        return data[data['Aquaculture System'].isin(system)]
    
    # Otherwise it's just 1 value
    else:
        return data[data['Aquaculture System'] == system]
    
def get_by_treatment(data: DataFrame, treatment: int):
    """
    Returns all the abundance samples of a given treatment (1 to 9)

    Args:
        data (DataFrame): The abundance dataframe
        treatment (int): The treatment number

    Raises:
        TypeError: data is not a DataFrame
        TypeError: treatment is not an int

    Returns:
        (DataFrame) the data grouped by system
    """

    if not isinstance(data, DataFrame): raise TypeError("data is not a DataFrame")
    if not isinstance(treatment, int): raise TypeError("treatment is not an int")

    return data[data['Treatment'] == treatment]

def get_by_timestep(data: DataFrame, timestep: float):
    """
    Returns all the abundance samples of a given timestep 

    Args:
        data (DataFrame): The abundance dataframe
        timestep (float): The timestep number

    Raises:
        TypeError: data is not a DataFrame
        TypeError: timestep is not a float

    Returns:
        (DataFrame) the data grouped by system
    """

    if not isinstance(data, DataFrame): raise TypeError("data is not a DataFrame")
    if not isinstance(timestep, float): raise TypeError("timestep is not an float")

    return data[data['Time Step'] == timestep]

def get_replicates(data: DataFrame, system: str, treatment: int, timestep: float):
    """
    Returns all the abundance replicate samples

    Args:
        data (DataFrame): The abundance dataframe
        timestep (float): The timestep number

    Raises:
        TypeError: data is not a DataFrame
        TypeError: timestep is not a float

    Returns:
        (DataFrame) the data grouped by system
    """

    if not isinstance(data, DataFrame): raise TypeError("data is not a DataFrame")
    if not isinstance(system, str): raise TypeError("system is not a string")
    if not isinstance(treatment, int): raise TypeError("treatment is not an int")
    if not isinstance(timestep, float): raise TypeError("timestep is not an float")

    # Mask to select replicates
    mask = (data["Aquaculture System"] == system) & (data["Treatment"] == treatment) & (data['Time Step'] == timestep)
    return data[mask]