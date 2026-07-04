make -f Makefile clean all
./classifierV32 <test.input >test.output2
diff test.output2 test.output > errors.txt

