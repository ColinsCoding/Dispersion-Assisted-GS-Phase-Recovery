-- comparator.vhd
-- One building block of a flash ADC: cmp_out = '1' iff v_in >= v_ref.
-- Uses VHDL's `real` type for v_in/v_ref -- valid for SIMULATION (this is
-- how you model an analog voltage feeding digital logic in a testbench)
-- but NOT synthesizable to real hardware as written; a real comparator
-- would be an analog transistor circuit, this is its behavioral stand-in.

library ieee;
use ieee.std_logic_1164.all;

entity comparator is
    port (
        v_in    : in  real;
        v_ref   : in  real;
        cmp_out : out std_logic
    );
end entity comparator;

architecture behavioral of comparator is
begin
    cmp_out <= '1' when v_in >= v_ref else '0';
end architecture behavioral;
