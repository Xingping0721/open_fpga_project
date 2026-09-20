// tb_ae_top.v
// Testbench for ae_top: applies the test vector from expected_vectors.vh
// and checks the outputs bit-exactly against the expected values.
`timescale 1ns / 1ps

module tb_ae_top;

    `include "expected_vectors.vh"

    reg  signed [15:0] x0, x1, x2, x3, x4, x5, x6, x7;
    wire signed [15:0] y0, y1, y2, y3, y4, y5, y6, y7;

    ae_top u_ae_top (
        .x0(x0), .x1(x1), .x2(x2), .x3(x3),
        .x4(x4), .x5(x5), .x6(x6), .x7(x7),
        .y0(y0), .y1(y1), .y2(y2), .y3(y3),
        .y4(y4), .y5(y5), .y6(y6), .y7(y7)
    );

    integer errors;

    task check(input [127:0] name, input signed [15:0] actual, input signed [15:0] expected);
        begin
            if (actual !== expected) begin
                $display("NG: %0s  actual=%0d (0x%h)  expected=%0d (0x%h)",
                          name, actual, actual, expected, expected);
                errors = errors + 1;
            end else begin
                $display("OK: %0s  actual=%0d (0x%h)", name, actual, actual);
            end
        end
    endtask

    initial begin
        errors = 0;

        x0 = TEST_X_0;
        x1 = TEST_X_1;
        x2 = TEST_X_2;
        x3 = TEST_X_3;
        x4 = TEST_X_4;
        x5 = TEST_X_5;
        x6 = TEST_X_6;
        x7 = TEST_X_7;

        #10;

        check("y0", y0, EXPECTED_Y_0);
        check("y1", y1, EXPECTED_Y_1);
        check("y2", y2, EXPECTED_Y_2);
        check("y3", y3, EXPECTED_Y_3);
        check("y4", y4, EXPECTED_Y_4);
        check("y5", y5, EXPECTED_Y_5);
        check("y6", y6, EXPECTED_Y_6);
        check("y7", y7, EXPECTED_Y_7);

        if (errors == 0) begin
            $display("ALL PASS");
        end else begin
            $display("%0d mismatch(es) found", errors);
        end

        $finish;
    end

endmodule
