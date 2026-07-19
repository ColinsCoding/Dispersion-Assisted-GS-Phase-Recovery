-- flash_adc_behavioral.vhd
-- A 2-bit (4-level) flash ADC, BEHAVIORAL style: describes WHAT the
-- circuit computes (an algorithm -- "which quarter of v_ref is v_in in")
-- with no reference to how it would actually be built from comparators.
-- This is the SEMANTICS of the ADC: the input/output mapping, full stop.

library ieee;
use ieee.std_logic_1164.all;

entity flash_adc_behavioral is
    port (
        v_in  : in  real;
        v_ref : in  real;
        code  : out std_logic_vector(1 downto 0)
    );
end entity flash_adc_behavioral;

architecture behavioral of flash_adc_behavioral is
begin
    process (v_in, v_ref)
    begin
        if v_in >= 0.75 * v_ref then
            code <= "11";
        elsif v_in >= 0.5 * v_ref then
            code <= "10";
        elsif v_in >= 0.25 * v_ref then
            code <= "01";
        else
            code <= "00";
        end if;
    end process;
end architecture behavioral;
