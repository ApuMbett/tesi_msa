class Agent:
    def __init__(self, api_manager, name, explainable: bool = False):
        self.api_manager = api_manager
        self.name = name 
        self.explainable = explainable
    def get_system_prompt(self):
        """
        Finds the system prompt for the agent. 
        Appends '_explainable' to the filename if the flag is True.
        """
        # Determine if we need the explainable suffix
        suffix = "_explainable" if self.explainable else ""
        
        # Dynamically construct the file path
        filepath = f"system_prompts/{self.name}{suffix}.txt"
        
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()