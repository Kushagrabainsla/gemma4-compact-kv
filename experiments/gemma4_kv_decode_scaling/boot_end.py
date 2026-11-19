"""Print the Unix time when the boot watchdog stops the Pod: the last "armed cap" line of its log plus the cap."""
import calendar
import sys
import time

end = ""
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) >= 4 and p[1] == "armed" and p[2] == "cap":
        end = calendar.timegm(time.strptime(p[0], "%Y-%m-%dT%H:%M:%SZ")) + int(p[3])
print(end)
