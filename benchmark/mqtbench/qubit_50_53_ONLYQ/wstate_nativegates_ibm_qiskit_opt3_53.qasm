OPENQASM 2.0;
include "qelib1.inc";
qreg q[53];
creg meas[53];
sx q[0];
rz(pi/4) q[0];
sx q[0];
sx q[1];
rz(0.6154797086703874) q[1];
sx q[1];
sx q[2];
rz(pi/6) q[2];
sx q[2];
sx q[3];
rz(0.46364760900080615) q[3];
sx q[3];
sx q[4];
rz(0.42053433528396456) q[4];
sx q[4];
sx q[5];
rz(0.38759668665518054) q[5];
sx q[5];
sx q[6];
rz(0.361367123906708) q[6];
sx q[6];
sx q[7];
rz(0.33983690945412137) q[7];
sx q[7];
sx q[8];
rz(0.32175055439664213) q[8];
sx q[8];
sx q[9];
rz(0.3062773691696692) q[9];
sx q[9];
sx q[10];
rz(0.29284277172857553) q[10];
sx q[10];
sx q[11];
rz(0.2810349015028142) q[11];
sx q[11];
sx q[12];
rz(0.2705497629785727) q[12];
sx q[12];
sx q[13];
rz(0.2611574109030239) q[13];
sx q[13];
sx q[14];
rz(0.25268025514207837) q[14];
sx q[14];
sx q[15];
rz(0.24497866312686423) q[15];
sx q[15];
sx q[16];
rz(0.23794112483020813) q[16];
sx q[16];
sx q[17];
rz(0.23147736397017793) q[17];
sx q[17];
sx q[18];
rz(0.22551340589813051) q[18];
sx q[18];
sx q[19];
rz(0.2199879773954594) q[19];
sx q[19];
sx q[20];
rz(0.21484983307571204) q[20];
sx q[20];
sx q[21];
rz(0.21005573907123853) q[21];
sx q[21];
sx q[22];
rz(0.20556893116117347) q[22];
sx q[22];
sx q[23];
rz(0.20135792079033044) q[23];
sx q[23];
sx q[24];
rz(0.19739555984988044) q[24];
sx q[24];
sx q[25];
rz(0.19365830044432641) q[25];
sx q[25];
sx q[26];
rz(0.19012560334646622) q[26];
sx q[26];
sx q[27];
rz(0.18677946108159382) q[27];
sx q[27];
sx q[28];
rz(0.18360401027891848) q[28];
sx q[28];
sx q[29];
rz(0.18058521419069784) q[29];
sx q[29];
sx q[30];
rz(0.1777106008451117) q[30];
sx q[30];
sx q[31];
rz(0.1749690456656885) q[31];
sx q[31];
sx q[32];
rz(0.17235058989932295) q[32];
sx q[32];
sx q[33];
rz(0.16984628808367352) q[33];
sx q[33];
sx q[34];
rz(0.1674480792196893) q[34];
sx q[34];
sx q[35];
rz(0.1651486774146269) q[35];
sx q[35];
sx q[36];
rz(0.16294147861051922) q[36];
sx q[36];
sx q[37];
rz(0.16082048067446486) q[37];
sx q[37];
sx q[38];
rz(0.15878021464576042) q[38];
sx q[38];
sx q[39];
rz(0.1568156853444007) q[39];
sx q[39];
sx q[40];
rz(0.15492231987081295) q[40];
sx q[40];
sx q[41];
rz(0.15309592278685402) q[41];
sx q[41];
sx q[42];
rz(0.15133263697721544) q[42];
sx q[42];
sx q[43];
rz(0.14962890935951734) q[43];
sx q[43];
sx q[44];
rz(0.1479814607487837) q[44];
sx q[44];
sx q[45];
rz(0.14638725929424856) q[45];
sx q[45];
sx q[46];
rz(0.14484349699855947) q[46];
sx q[46];
sx q[47];
rz(0.14334756890536537) q[47];
sx q[47];
sx q[48];
rz(0.14189705460416402) q[48];
sx q[48];
sx q[49];
rz(0.14048970175352027) q[49];
sx q[49];
sx q[50];
rz(0.13912341136739848) q[50];
sx q[50];
sx q[51];
rz(0.13779622464588526) q[51];
sx q[51];
x q[52];
cx q[52],q[51];
sx q[51];
rz(0.1377962246458848) q[51];
sx q[51];
cx q[51],q[50];
sx q[50];
rz(0.13912341136739848) q[50];
sx q[50];
cx q[50],q[49];
sx q[49];
rz(0.14048970175351982) q[49];
sx q[49];
cx q[49],q[48];
sx q[48];
rz(0.14189705460416402) q[48];
sx q[48];
cx q[48],q[47];
sx q[47];
rz(0.14334756890536582) q[47];
sx q[47];
cx q[47],q[46];
sx q[46];
rz(0.14484349699855903) q[46];
sx q[46];
cx q[46],q[45];
sx q[45];
rz(0.146387259294249) q[45];
sx q[45];
cx q[45],q[44];
sx q[44];
rz(0.1479814607487837) q[44];
sx q[44];
cx q[44],q[43];
sx q[43];
rz(0.14962890935951734) q[43];
sx q[43];
cx q[43],q[42];
sx q[42];
rz(0.15133263697721588) q[42];
sx q[42];
cx q[42],q[41];
sx q[41];
rz(0.15309592278685447) q[41];
sx q[41];
cx q[41],q[40];
sx q[40];
rz(0.1549223198708134) q[40];
sx q[40];
cx q[40],q[39];
sx q[39];
rz(0.1568156853444007) q[39];
sx q[39];
cx q[39],q[38];
sx q[38];
rz(0.15878021464576086) q[38];
sx q[38];
cx q[38],q[37];
sx q[37];
rz(0.16082048067446486) q[37];
sx q[37];
cx q[37],q[36];
sx q[36];
rz(0.16294147861051922) q[36];
sx q[36];
cx q[36],q[35];
sx q[35];
rz(0.1651486774146269) q[35];
sx q[35];
cx q[35],q[34];
sx q[34];
rz(0.16744807921968885) q[34];
sx q[34];
cx q[34],q[33];
sx q[33];
rz(0.16984628808367397) q[33];
sx q[33];
cx q[33],q[32];
sx q[32];
rz(0.17235058989932295) q[32];
sx q[32];
cx q[32],q[31];
sx q[31];
rz(0.1749690456656885) q[31];
sx q[31];
cx q[31],q[30];
sx q[30];
rz(0.1777106008451117) q[30];
sx q[30];
cx q[30],q[29];
sx q[29];
rz(0.18058521419069784) q[29];
sx q[29];
cx q[29],q[28];
sx q[28];
rz(0.18360401027891804) q[28];
sx q[28];
cx q[28],q[27];
sx q[27];
rz(0.18677946108159382) q[27];
sx q[27];
cx q[27],q[26];
sx q[26];
rz(0.19012560334646622) q[26];
sx q[26];
cx q[26],q[25];
sx q[25];
rz(0.19365830044432641) q[25];
sx q[25];
cx q[25],q[24];
sx q[24];
rz(0.19739555984988044) q[24];
sx q[24];
cx q[24],q[23];
sx q[23];
rz(0.20135792079033088) q[23];
sx q[23];
cx q[23],q[22];
sx q[22];
rz(0.20556893116117436) q[22];
sx q[22];
cx q[22],q[21];
sx q[21];
rz(0.21005573907123853) q[21];
sx q[21];
cx q[21],q[20];
sx q[20];
rz(0.21484983307571248) q[20];
sx q[20];
cx q[20],q[19];
sx q[19];
rz(0.2199879773954594) q[19];
sx q[19];
cx q[19],q[18];
sx q[18];
rz(0.2255134058981314) q[18];
sx q[18];
cx q[18],q[17];
sx q[17];
rz(0.23147736397017837) q[17];
sx q[17];
cx q[17],q[16];
sx q[16];
rz(0.23794112483020857) q[16];
sx q[16];
cx q[16],q[15];
sx q[15];
rz(0.24497866312686423) q[15];
sx q[15];
cx q[15],q[14];
sx q[14];
rz(0.2526802551420788) q[14];
sx q[14];
cx q[14],q[13];
sx q[13];
rz(0.26115741090302436) q[13];
sx q[13];
cx q[13],q[12];
sx q[12];
rz(0.2705497629785727) q[12];
sx q[12];
cx q[12],q[11];
sx q[11];
rz(0.28103490150281374) q[11];
sx q[11];
cx q[11],q[10];
sx q[10];
rz(0.29284277172857553) q[10];
sx q[10];
cx q[10],q[9];
sx q[9];
rz(0.30627736916966963) q[9];
sx q[9];
cx q[9],q[8];
sx q[8];
rz(0.32175055439664213) q[8];
sx q[8];
cx q[8],q[7];
sx q[7];
rz(0.3398369094541218) q[7];
sx q[7];
cx q[7],q[6];
sx q[6];
rz(0.36136712390670755) q[6];
sx q[6];
cx q[6],q[5];
sx q[5];
rz(0.387596686655181) q[5];
sx q[5];
cx q[5],q[4];
sx q[4];
rz(0.420534335283965) q[4];
sx q[4];
cx q[4],q[3];
sx q[3];
rz(0.46364760900080615) q[3];
sx q[3];
cx q[3],q[2];
sx q[2];
rz(pi/6) q[2];
sx q[2];
cx q[2],q[1];
sx q[1];
rz(0.6154797086703869) q[1];
sx q[1];
cx q[1],q[0];
sx q[0];
rz(pi/4) q[0];
sx q[0];
cx q[51],q[52];
cx q[50],q[51];
cx q[49],q[50];
cx q[48],q[49];
cx q[47],q[48];
cx q[46],q[47];
cx q[45],q[46];
cx q[44],q[45];
cx q[43],q[44];
cx q[42],q[43];
cx q[41],q[42];
cx q[40],q[41];
cx q[39],q[40];
cx q[38],q[39];
cx q[37],q[38];
cx q[36],q[37];
cx q[35],q[36];
cx q[34],q[35];
cx q[33],q[34];
cx q[32],q[33];
cx q[31],q[32];
cx q[30],q[31];
cx q[29],q[30];
cx q[28],q[29];
cx q[27],q[28];
cx q[26],q[27];
cx q[25],q[26];
cx q[24],q[25];
cx q[23],q[24];
cx q[22],q[23];
cx q[21],q[22];
cx q[20],q[21];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[18];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[15];
cx q[13],q[14];
cx q[12],q[13];
cx q[11],q[12];
cx q[10],q[11];
cx q[9],q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23],q[24],q[25],q[26],q[27],q[28],q[29],q[30],q[31],q[32],q[33],q[34],q[35],q[36],q[37],q[38],q[39],q[40],q[41],q[42],q[43],q[44],q[45],q[46],q[47],q[48],q[49],q[50],q[51],q[52];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];
measure q[17] -> meas[17];
measure q[18] -> meas[18];
measure q[19] -> meas[19];
measure q[20] -> meas[20];
measure q[21] -> meas[21];
measure q[22] -> meas[22];
measure q[23] -> meas[23];
measure q[24] -> meas[24];
measure q[25] -> meas[25];
measure q[26] -> meas[26];
measure q[27] -> meas[27];
measure q[28] -> meas[28];
measure q[29] -> meas[29];
measure q[30] -> meas[30];
measure q[31] -> meas[31];
measure q[32] -> meas[32];
measure q[33] -> meas[33];
measure q[34] -> meas[34];
measure q[35] -> meas[35];
measure q[36] -> meas[36];
measure q[37] -> meas[37];
measure q[38] -> meas[38];
measure q[39] -> meas[39];
measure q[40] -> meas[40];
measure q[41] -> meas[41];
measure q[42] -> meas[42];
measure q[43] -> meas[43];
measure q[44] -> meas[44];
measure q[45] -> meas[45];
measure q[46] -> meas[46];
measure q[47] -> meas[47];
measure q[48] -> meas[48];
measure q[49] -> meas[49];
measure q[50] -> meas[50];
measure q[51] -> meas[51];
measure q[52] -> meas[52];
