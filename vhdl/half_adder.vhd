-- half_adder.vhd
-- Mirrors dgs/digital_logic.py:half_adder(a, b) -> (sum, carry) exactly:
--   sum   = a XOR b
--   carry = a AND b
-- This is the smallest possible bridge between the Python *behavioral* model
-- (an int function you can call from a REPL) and a *structural* hardware
-- description (gates wired with signals, synthesizable to an FPGA/ASIC).

library ieee;
use ieee.std_logic_1164.all;

entity half_adder is
    port (
        a     : in  std_logic;
        b     : in  std_logic;
        sum   : out std_logic;
        carry : out std_logic
    );
end entity half_adder;

architecture structural of half_adder is
begin
    sum   <= a xor b;
    carry <= a and b;
end architecture structural;
