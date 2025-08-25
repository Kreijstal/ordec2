* Simple RC circuit
V1 in 0 ac 1
R1 in out 1k
C1 out 0 1nF

.control
ac dec 10 1 1G
wrdata test_data.txt all
.endc

.end
