"""Stop this Pod with the Pod-scoped RUNPOD_API_KEY that Runpod sets (AUTO_STOP; RUNPOD.md Section 3). Prints the HTTP
status only, never the key."""
import json
import os
import urllib.request

q = {"query": 'mutation { podStop(input: {podId: "%s"}) { id } }' % os.environ["RUNPOD_POD_ID"]}
r = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps(q).encode(), method="POST",
                           headers={"Content-Type": "application/json",
                                    "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]})
print(urllib.request.urlopen(r, timeout=30).status)
