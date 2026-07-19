-- tb_ripple_carry_adder.vhd
-- Self-checking testbench: exhaustively exercises a 4-bit ripple_carry_adder
-- against the arithmetic answer computed directly in VHDL (to_integer/
-- to_unsigned), and reports PASS/FAIL per vector -- the VHDL-side counterpart
-- to the Python cross-check in scripts/check_vhdl_vs_python_logic.py, which
-- exhaustively checks the *same* 4-bit adder via dgs/digital_logic.py.
--
-- Requires a VHDL simulator (GHDL, ModelSim, Vivado xsim, ...) -- none is
-- installed in this dev environment, so this file is a reference design,
-- not something CI here has executed. To run with GHDL once installed:
--   ghdl -a half_adder.vhd full_adder.vhd ripple_carry_adder.vhd tb_ripple_carry_adder.vhd
--   ghdl -e tb_ripple_carry_adder
--   ghdl -r tb_ripple_carry_adder

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_ripple_carry_adder is
end entity tb_ripple_carry_adder;

architecture sim of tb_ripple_carry_adder is

    constant N : integer := 4;

    component ripple_carry_adder is
        generic (N : integer := 4);
        port (
            a, b  : in  std_logic_vector(N-1 downto 0);
            cin   : in  std_logic;
            sum   : out std_logic_vector(N-1 downto 0);
            cout  : out std_logic
        );
    end component;

    signal a_tb, b_tb, sum_tb : std_logic_vector(N-1 downto 0);
    signal cin_tb, cout_tb    : std_logic;

    signal n_checked, n_failed : integer := 0;

begin

    DUT: ripple_carry_adder
        generic map (N => N)
        port map (a => a_tb, b => b_tb, cin => cin_tb, sum => sum_tb, cout => cout_tb);

    STIMULUS: process
        variable expected : integer;
        variable actual    : integer;
    begin
        cin_tb <= '0';
        for ai in 0 to 2**N - 1 loop
            for bi in 0 to 2**N - 1 loop
                a_tb <= std_logic_vector(to_unsigned(ai, N));
                b_tb <= std_logic_vector(to_unsigned(bi, N));
                wait for 1 ns;

                expected := ai + bi;
                actual   := to_integer(unsigned(sum_tb));
                if cout_tb = '1' then
                    actual := actual + 2**N;
                end if;

                n_checked <= n_checked + 1;
                if actual /= expected then
                    n_failed <= n_failed + 1;
                    report "FAIL a=" & integer'image(ai) & " b=" & integer'image(bi)
                         & " expected=" & integer'image(expected)
                         & " actual=" & integer'image(actual)
                         severity error;
                end if;
            end loop;
        end loop;

        wait for 1 ns;
        if n_failed = 0 then
            report "PASS: all " & integer'image(n_checked) & " vectors matched a+b exactly";
        else
            report "FAIL: " & integer'image(n_failed) & " / " & integer'image(n_checked) & " vectors mismatched";
        end if;
        wait;
    end process STIMULUS;

end architecture sim;
