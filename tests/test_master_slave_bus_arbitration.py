"""Tests for dgs/master_slave_bus_arbitration.py"""
import pytest

from dgs.master_slave_bus_arbitration import (
    i2c_bus_arbitration, address_to_bits,
    spi_daisy_chain_shift, spi_chip_select_polling_time,
)


class TestAddressToBits:
    def test_known_value(self):
        # 0x2A = 0b0101010
        assert address_to_bits(0x2A, rw_bit=0) == [0, 1, 0, 1, 0, 1, 0, 0]

    def test_rw_bit_appended(self):
        assert address_to_bits(0x00, rw_bit=1)[-1] == 1

    def test_rejects_out_of_range_address(self):
        with pytest.raises(ValueError):
            address_to_bits(0x80)

    def test_rejects_bad_rw_bit(self):
        with pytest.raises(ValueError):
            address_to_bits(0x10, rw_bit=2)


class TestI2CArbitration:
    def test_lower_address_always_wins(self):
        # try several pairs; the wired-AND rule means the numerically
        # smaller address must win every time, not just in one example
        pairs = [(0x2A, 0x2C), (0x01, 0x7F), (0x10, 0x11), (0x00, 0x01), (0x55, 0x56)]
        for a, b in pairs:
            result = i2c_bus_arbitration({"A": address_to_bits(a), "B": address_to_bits(b)})
            expected = "A" if a < b else "B"
            assert result["winner"] == expected, f"addresses {a:#x} vs {b:#x}"

    def test_bus_is_wired_and_of_contenders(self):
        # while both masters still contend, the bus bit must be the AND
        # of their intended bits at that position
        bits_A = [0, 1, 1, 0]
        bits_B = [0, 1, 0, 1]
        result = i2c_bus_arbitration({"A": bits_A, "B": bits_B})
        # bit 0: AND(0,0)=0 ; bit 1: AND(1,1)=1 ; bit 2: A loses here (A=1,B=0->bus=0)
        assert result["bus_sequence"][:3] == [0, 1, 0]

    def test_loser_recorded_at_correct_bit_index(self):
        result = i2c_bus_arbitration({"A": [0, 1, 0, 1, 0], "B": [0, 1, 0, 1, 1]})
        assert result["losers"] == [("B", 4)]
        assert result["winner"] == "A"

    def test_identical_addresses_is_a_true_tie(self):
        same = [1, 0, 1, 0, 1, 0, 1]
        result = i2c_bus_arbitration({"A": same, "B": same})
        assert result["winner"] is None
        assert result["losers"] == []

    def test_requires_at_least_two_masters(self):
        with pytest.raises(ValueError):
            i2c_bus_arbitration({"A": [0, 1, 0]})

    def test_requires_equal_length_bitstreams(self):
        with pytest.raises(ValueError):
            i2c_bus_arbitration({"A": [0, 1], "B": [0, 1, 1]})

    def test_three_way_arbitration(self):
        # C has the lowest address and must win even with two competitors
        result = i2c_bus_arbitration({
            "A": address_to_bits(0x50), "B": address_to_bits(0x40), "C": address_to_bits(0x30),
        })
        assert result["winner"] == "C"


class TestSPIDaisyChain:
    def test_full_shift_exactly_replaces_register_with_master_data(self):
        # the defining identity of a full N*8-clock daisy-chain shift:
        # after it, the register holds EXACTLY what the master pushed in,
        # in the same order -- checked directly, not just "looks plausible"
        slaves_before = [0xAA, 0xBB, 0xCC]
        master_out = [0x11, 0x22, 0x33]
        result = spi_daisy_chain_shift(slaves_before, master_out)
        assert result["new_register_contents"] == master_out

    def test_full_shift_outputs_original_contents_on_miso_in_order(self):
        slaves_before = [0xAA, 0xBB, 0xCC]
        master_out = [0x11, 0x22, 0x33]
        result = spi_daisy_chain_shift(slaves_before, master_out)
        assert result["miso_output_bytes"] == slaves_before

    def test_clocks_required_is_n_times_8(self):
        result = spi_daisy_chain_shift([0x00, 0x00, 0x00, 0x00])
        assert result["clocks_required"] == 32

    def test_single_slave_is_a_simple_swap(self):
        result = spi_daisy_chain_shift([0x5A], [0xA5])
        assert result["new_register_contents"] == [0xA5]
        assert result["miso_output_bytes"] == [0x5A]

    def test_default_master_out_is_zero(self):
        result = spi_daisy_chain_shift([0xFF, 0xFF])
        assert result["new_register_contents"] == [0x00, 0x00]

    def test_rejects_empty_chain(self):
        with pytest.raises(ValueError):
            spi_daisy_chain_shift([])

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            spi_daisy_chain_shift([0x11, 0x22], [0x33])

    def test_rejects_out_of_range_byte(self):
        with pytest.raises(ValueError):
            spi_daisy_chain_shift([0x100])


class TestSPIChipSelectPolling:
    def test_scales_linearly_with_slave_count(self):
        one = spi_chip_select_polling_time(1, 16, 10e6)
        eight = spi_chip_select_polling_time(8, 16, 10e6)
        assert eight["total_seconds_to_poll_all"] == pytest.approx(8 * one["total_seconds_to_poll_all"])

    def test_faster_clock_reduces_transfer_time(self):
        slow = spi_chip_select_polling_time(4, 16, 1e6)
        fast = spi_chip_select_polling_time(4, 16, 10e6)
        assert fast["total_seconds_to_poll_all"] < slow["total_seconds_to_poll_all"]

    def test_rejects_nonpositive_inputs(self):
        with pytest.raises(ValueError):
            spi_chip_select_polling_time(0, 16, 10e6)
        with pytest.raises(ValueError):
            spi_chip_select_polling_time(4, 0, 10e6)
        with pytest.raises(ValueError):
            spi_chip_select_polling_time(4, 16, 0)
