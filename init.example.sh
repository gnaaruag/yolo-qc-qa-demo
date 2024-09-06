
#!/bin/bash

# Check if the 'uploads' directory exists
if [ ! -d "uploads" ]; then
	# Create the 'uploads' directory
	mkdir uploads
fi

# Export the REPLICATE_API_KEY environment variable
export REPLICATE_API_KEY="some_api"
