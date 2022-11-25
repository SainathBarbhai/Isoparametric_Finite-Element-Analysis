clear all; close all; clc

format long

%initial parameters-->
X= 10; %mm units
Y= 5; %mm units
e=1000; %youngs modulus in N/mm2
mu= 0.3; %possion ratio
N= 5;

E= (e/((1+mu)*(1-2*mu)))*[1-mu mu 0; mu 1-mu 0; 0 0 (1-2*mu)/2];

% connectivity matrix--> anti-clock indexing
con=zeros(8,N);
con(1,1)= 1; con(2,1)= 2; con(3,1)=3; con(4,1)= 4; con(5,1)= 11; con(6,1)= 12; con(7,1)=9; con(8,1)= 10;
con(1,2)= 3; con(2,2)= 4; con(3,2)=5; con(4,2)= 6; con(5,2)= 13; con(6,2)= 14; con(7,2)= 11; con(8,2)= 12;
con(1,3)= 5; con(2,3)= 6; con(3,3)=7; con(4,3)= 8; con(5,3)= 19; con(6,3)= 20; con(7,3)=13; con(8,3)= 14;
con(1,4)= 9; con(2,4)= 10; con(3,4)=11; con(4,4)= 12; con(5,4)= 17; con(6,4)= 18; con(7,4)=15; con(8,4)= 16;
con(1,5)= 11; con(2,5)= 12; con(3,5)=13; con(4,5)= 14; con(5,5)= 19; con(6,5)= 20; con(7,5)=17; con(8,5)= 18;


%coordinate matrix--> (units mm)
x=zeros(N-1,1);
y=zeros(N-1,1);

x(1)=0; y(1)=0;
x(2)=3.33; y(2)=0;
x(3)=6.66; y(3)=0;
x(4)=10; y(4)=0;
x(5)=0; y(5)=2.5;
x(6)=3.33; y(6)=2.5;
x(7)=6.66; y(7)=2.5;
x(8)=0; y(8)=5;
x(9)=3.33; y(9)=5;
x(10)=10; y(10)=5;

%creating local k matrix-->
%for element 1-->
J1= [1.665 0; 0 1.25];  %derived Jacobian for each element

%partial derivative of shape function wrto eta and zeta
% Shape_Func_ZE= [(1+eta)/4 (-1-eta)/4 (-1+eta)/4 (1-eta)/4; (1+zeta)/4 (1-zeta)/4 (-1+zeta)/4 (-1-zeta)/4 ];

%all Shape_Func_ZE are constant throughout-->
Shape_Func_ZE_1= [(1+1/sqrt(3))/4 (-1-1/sqrt(3))/4 (-1+1/sqrt(3))/4 (1-1/sqrt(3))/4; (1+1/sqrt(3))/4 (1-1/sqrt(3))/4 (-1+1/sqrt(3))/4 (-1-1/sqrt(3))/4];
Shape_Func_ZE_2= [(1+1/sqrt(3))/4 (-1-1/sqrt(3))/4 (-1+1/sqrt(3))/4 (1-1/sqrt(3))/4; (1-1/sqrt(3))/4 (1+1/sqrt(3))/4 (-1-1/sqrt(3))/4 (-1+1/sqrt(3))/4];
Shape_Func_ZE_3= [(1-1/sqrt(3))/4 (-1+1/sqrt(3))/4 (-1-1/sqrt(3))/4 (1+1/sqrt(3))/4; (1-1/sqrt(3))/4 (1+1/sqrt(3))/4 (-1-1/sqrt(3))/4 (-1+1/sqrt(3))/4];
Shape_Func_ZE_4= [(1-1/sqrt(3))/4 (-1+1/sqrt(3))/4 (-1-1/sqrt(3))/4 (1+1/sqrt(3))/4; (1+1/sqrt(3))/4 (1-1/sqrt(3))/4 (-1+1/sqrt(3))/4 (-1-1/sqrt(3))/4];


Shape_Func_xy_11= inv(J1)*Shape_Func_ZE_1; %Shape_Func partial derivative interms of x and y
Shape_Func_xy_12= inv(J1)*Shape_Func_ZE_2;
Shape_Func_xy_13= inv(J1)*Shape_Func_ZE_3;
Shape_Func_xy_14= inv(J1)*Shape_Func_ZE_4;

for i=1:4
    A11(:,:,i)= [Shape_Func_xy_11(1,i) 0; 0 Shape_Func_xy_11(2,i); Shape_Func_xy_11(2,i) Shape_Func_xy_11(1,i)];  
    A12(:,:,i)= [Shape_Func_xy_12(1,i) 0; 0 Shape_Func_xy_12(2,i); Shape_Func_xy_12(2,i) Shape_Func_xy_12(1,i)];
    A13(:,:,i)= [Shape_Func_xy_13(1,i) 0; 0 Shape_Func_xy_13(2,i); Shape_Func_xy_13(2,i) Shape_Func_xy_13(1,i)];
    A14(:,:,i)= [Shape_Func_xy_14(1,i) 0; 0 Shape_Func_xy_14(2,i); Shape_Func_xy_14(2,i) Shape_Func_xy_14(1,i)];
end

B11=[A11(:,:,1) A11(:,:,2) A11(:,:,3) A11(:,:,4)];
B12=[A12(:,:,1) A12(:,:,2) A12(:,:,3) A12(:,:,4)];
B13=[A13(:,:,1) A13(:,:,2) A13(:,:,3) A13(:,:,4)];
B14=[A14(:,:,1) A14(:,:,2) A14(:,:,3) A14(:,:,4)];

% implimenting gauss quadrature intergration for getting k1 local-->
k11= transpose(B11)*E*B11*det(J1); 
k12= transpose(B12)*E*B12*det(J1);
k13= transpose(B13)*E*B13*det(J1);
k14= transpose(B14)*E*B14*det(J1);  %local k1 multiplied by 4 for gauss quadrature

k1= k11 + k12 + k13 + k14;

% r1= 4*E*Shape_Func_xy_1*det(J1);   %local r1 multiplied by 4 for gauss quadrature


%for element 2-->
J2= [1.665 0; 0 1.25];
%implimenting gauss quadrature intergration for getting k2 local-->

Shape_Func_xy_21= inv(J2)*Shape_Func_ZE_1; %Shape_Func partial derivative interms of x and y
Shape_Func_xy_22= inv(J2)*Shape_Func_ZE_2;
Shape_Func_xy_23= inv(J2)*Shape_Func_ZE_3;
Shape_Func_xy_24= inv(J2)*Shape_Func_ZE_4;

for i=1:4
    A21(:,:,i)= [Shape_Func_xy_21(1,i) 0; 0 Shape_Func_xy_21(2,i); Shape_Func_xy_21(2,i) Shape_Func_xy_21(1,i)];  
    A22(:,:,i)= [Shape_Func_xy_22(1,i) 0; 0 Shape_Func_xy_22(2,i); Shape_Func_xy_22(2,i) Shape_Func_xy_22(1,i)];
    A23(:,:,i)= [Shape_Func_xy_23(1,i) 0; 0 Shape_Func_xy_23(2,i); Shape_Func_xy_23(2,i) Shape_Func_xy_23(1,i)];
    A24(:,:,i)= [Shape_Func_xy_24(1,i) 0; 0 Shape_Func_xy_24(2,i); Shape_Func_xy_24(2,i) Shape_Func_xy_24(1,i)];
end

B21=[A21(:,:,1) A21(:,:,2) A21(:,:,3) A21(:,:,4)];
B22=[A22(:,:,1) A22(:,:,2) A22(:,:,3) A22(:,:,4)];
B23=[A23(:,:,1) A23(:,:,2) A23(:,:,3) A23(:,:,4)];
B24=[A24(:,:,1) A24(:,:,2) A24(:,:,3) A24(:,:,4)];


k2_1= transpose(B21)*E* B21* det(J2);
k2_2= transpose(B22)*E* B22* det(J2);
k2_3= transpose(B23)*E* B23* det(J2);
k2_4= transpose(B24)*E* B24* det(J2);

k2= k2_1+ k2_2+ k2_3+ k2_4;  %local k2

% r2= E*Shape_Func_xy*det(J2_1)+ E*Shape_Func_xy*det(J2_2)+E*Shape_Func_xy*det(J2_3)+E*Shape_Func_xy*det(J2_4);  %local r2

%for element 3-->
J3_1= [1.67 (2.5/4)*(-1-1/sqrt(3))+(5/4)*(1+1/sqrt(3)); 0 (2.5/4)*(1-1/sqrt(3))+(5/4)*(1+1/sqrt(3))];    %jacobian with all 4 points for gauss quadrature
J3_2= [1.67 (2.5/4)*(-1+1/sqrt(3))+(5/4)*(1-1/sqrt(3)); 0 (2.5/4)*(1-1/sqrt(3))+(5/4)*(1+1/sqrt(3))];
J3_3= [1.67 (2.5/4)*(-1+1/sqrt(3))+(5/4)*(1-1/sqrt(3)); 0 (2.5/4)*(1+1/sqrt(3))+(5/4)*(1-1/sqrt(3))];
J3_4= [1.67 (2.5/4)*(-1-1/sqrt(3))+(5/4)*(1+1/sqrt(3)); 0 (2.5/4)*(1+1/sqrt(3))+(5/4)*(1-1/sqrt(3))];

%implimenting gauss quadrature intergration for getting k3 local-->

Shape_Func_xy_31= inv(J3_1)*Shape_Func_ZE_1; %Shape_Func partial derivative interms of x and y
Shape_Func_xy_32= inv(J3_2)*Shape_Func_ZE_2;
Shape_Func_xy_33= inv(J3_3)*Shape_Func_ZE_3;
Shape_Func_xy_34= inv(J3_4)*Shape_Func_ZE_4;

for i=1:4
    A31(:,:,i)= [Shape_Func_xy_31(1,i) 0; 0 Shape_Func_xy_31(2,i); Shape_Func_xy_31(2,i) Shape_Func_xy_31(1,i)];  
    A32(:,:,i)= [Shape_Func_xy_32(1,i) 0; 0 Shape_Func_xy_32(2,i); Shape_Func_xy_32(2,i) Shape_Func_xy_32(1,i)];
    A33(:,:,i)= [Shape_Func_xy_33(1,i) 0; 0 Shape_Func_xy_33(2,i); Shape_Func_xy_33(2,i) Shape_Func_xy_33(1,i)];
    A34(:,:,i)= [Shape_Func_xy_34(1,i) 0; 0 Shape_Func_xy_34(2,i); Shape_Func_xy_34(2,i) Shape_Func_xy_34(1,i)];
end

B31=[A31(:,:,1) A31(:,:,2) A31(:,:,3) A31(:,:,4)];
B32=[A32(:,:,1) A32(:,:,2) A32(:,:,3) A32(:,:,4)];
B33=[A33(:,:,1) A33(:,:,2) A33(:,:,3) A33(:,:,4)];
B34=[A34(:,:,1) A34(:,:,2) A34(:,:,3) A34(:,:,4)];


k3_1= transpose(B31)*E* B31* det(J3_1);
k3_2= transpose(B32)*E* B32* det(J3_2);
k3_3= transpose(B33)*E* B33* det(J3_3);
k3_4= transpose(B34)*E* B34* det(J3_4);

k3= k3_1+ k3_2+ k3_3+ k3_4;   %local k3

% r3= E*Shape_Func_xy*det(J3_1)+ E*Shape_Func_xy*det(J3_2)+E*Shape_Func_xy*det(J3_3)+E*Shape_Func_xy*det(J3_4);   %local r3

%for element 4-->
J4= [1.665 0; 0 1.25];

Shape_Func_xy_41= inv(J4)*Shape_Func_ZE_1; %Shape_Func partial derivative interms of x and y
Shape_Func_xy_42= inv(J4)*Shape_Func_ZE_2;
Shape_Func_xy_43= inv(J4)*Shape_Func_ZE_3;
Shape_Func_xy_44= inv(J4)*Shape_Func_ZE_4;

for i=1:4
    A41(:,:,i)= [Shape_Func_xy_41(1,i) 0; 0 Shape_Func_xy_41(2,i); Shape_Func_xy_41(2,i) Shape_Func_xy_41(1,i)];  
    A42(:,:,i)= [Shape_Func_xy_42(1,i) 0; 0 Shape_Func_xy_42(2,i); Shape_Func_xy_42(2,i) Shape_Func_xy_42(1,i)];
    A43(:,:,i)= [Shape_Func_xy_43(1,i) 0; 0 Shape_Func_xy_43(2,i); Shape_Func_xy_43(2,i) Shape_Func_xy_43(1,i)];
    A44(:,:,i)= [Shape_Func_xy_44(1,i) 0; 0 Shape_Func_xy_44(2,i); Shape_Func_xy_44(2,i) Shape_Func_xy_44(1,i)];
end

B41=[A41(:,:,1) A41(:,:,2) A41(:,:,3) A41(:,:,4)];
B42=[A42(:,:,1) A42(:,:,2) A42(:,:,3) A42(:,:,4)];
B43=[A43(:,:,1) A43(:,:,2) A43(:,:,3) A43(:,:,4)];
B44=[A44(:,:,1) A44(:,:,2) A44(:,:,3) A44(:,:,4)];

% implimenting gauss quadrature intergration for getting k1 local-->
k41= transpose(B41)*E*B41*det(J4); 
k42= transpose(B42)*E*B42*det(J4);
k43= transpose(B43)*E*B43*det(J4);
k44= transpose(B44)*E*B44*det(J4);  %local k1 multiplied by 4 for gauss quadrature

k4= k41 + k42 + k43 + k44;

% r4= 4*E*Shape_Func_xy*det(J4);   %local r4 multiplied by 4 for gauss quadrature


%for element 5-->
% syms eta zeta
J5_1= [-1.665+(6.66/4)*(1-1/sqrt(3))+(10/4)*(1+1/sqrt(3)) 0; (6.66/4)*(-1-1/sqrt(3))+(10/4)*(1+1/sqrt(3)) 1.25 ];    %jacobian with all 4 points for gauss quadrature
J5_2= [-1.665+(6.66/4)*(1+1/sqrt(3))+(10/4)*(1-1/sqrt(3)) 0; (6.66/4)*(-1-1/sqrt(3))+(10/4)*(1+1/sqrt(3)) 1.25 ];
J5_3= [-1.665+(6.66/4)*(1+1/sqrt(3))+(10/4)*(1-1/sqrt(3)) 0; (6.66/4)*(-1+1/sqrt(3))+(10/4)*(1-1/sqrt(3)) 1.25 ];
J5_4= [-1.665+(6.66/4)*(1-1/sqrt(3))+(10/4)*(1+1/sqrt(3)) 0; (6.66/4)*(-1+1/sqrt(3))+(10/4)*(1-1/sqrt(3)) 1.25 ];

%implimenting gauss quadrature intergration for getting k5 local-->
Shape_Func_xy_51= inv(J5_1)*Shape_Func_ZE_1; %Shape_Func partial derivative interms of x and y
Shape_Func_xy_52= inv(J5_2)*Shape_Func_ZE_2;
Shape_Func_xy_53= inv(J5_3)*Shape_Func_ZE_3;
Shape_Func_xy_54= inv(J5_4)*Shape_Func_ZE_4;

for i=1:4
    A51(:,:,i)= [Shape_Func_xy_51(1,i) 0; 0 Shape_Func_xy_51(2,i); Shape_Func_xy_51(2,i) Shape_Func_xy_51(1,i)];  
    A52(:,:,i)= [Shape_Func_xy_52(1,i) 0; 0 Shape_Func_xy_52(2,i); Shape_Func_xy_52(2,i) Shape_Func_xy_52(1,i)];
    A53(:,:,i)= [Shape_Func_xy_53(1,i) 0; 0 Shape_Func_xy_53(2,i); Shape_Func_xy_53(2,i) Shape_Func_xy_53(1,i)];
    A54(:,:,i)= [Shape_Func_xy_54(1,i) 0; 0 Shape_Func_xy_54(2,i); Shape_Func_xy_54(2,i) Shape_Func_xy_54(1,i)];
end

B51=[A51(:,:,1) A51(:,:,2) A51(:,:,3) A51(:,:,4)];
B52=[A52(:,:,1) A52(:,:,2) A52(:,:,3) A52(:,:,4)];
B53=[A53(:,:,1) A53(:,:,2) A53(:,:,3) A53(:,:,4)];
B54=[A54(:,:,1) A54(:,:,2) A54(:,:,3) A54(:,:,4)];


k5_1= transpose(B51)*E* B51* det(J5_1);
k5_2= transpose(B52)*E* B52* det(J5_2);
k5_3= transpose(B53)*E* B53* det(J5_3);
k5_4= transpose(B54)*E* B54* det(J5_4);

k5= k5_1+ k5_2+ k5_3+ k5_4 ;     %local k5

% % r5= E*Shape_Func_xy*det(J5_1)+ E*Shape_Func_xy*det(J5_2)+ E*Shape_Func_xy*det(J5_3)+ E*Shape_Func_xy*det(J5_4);    %local r5


% creating global k matrix-->

K= zeros((N)*4);  %initializing K Global matrix
R= zeros(N*4,2);  %initializing R Global vector

 for i = 1:8

    for j=1:8  %looping over row and column 
        K(con(i,1),con(j,1))= K(con(i,1),con(j,1)) + k1(i,j);
        K(con(i,2),con(j,2))= K(con(i,2),con(j,2)) + k2(i,j);
        K(con(i,3),con(j,3))= K(con(i,3),con(j,3)) + k3(i,j);
        K(con(i,4),con(j,4))= K(con(i,4),con(j,4)) + k4(i,j);
        K(con(i,5),con(j,5))= K(con(i,5),con(j,5)) + k5(i,j);

    end

 end
K_complete= K;

K(1,:)=[]; % imposing dof boundary conditions and truncating respective rows and columns of GLOBAL K
K(:,1)=[];
K(1,:)=[];
K(:,1)=[];
K(2,:)=[];
K(:,2)=[];
K(3,:)=[];
K(:,3)=[];
K(4,:)=[];
K(:,4)=[];
K(4,:)=[];
K(:,4)=[];
K(9,:)=[];
K(:,9)=[];

Global_Force_Vector= [0; 0; 1.8e3; 0; 0; 0; 0; 0; 16.65e3; 0; 49.95e3; 4.2e3; 33.3e3]; %Forces in Newtons

Displacement= inv(K)*Global_Force_Vector; % u2(1) u3(2) u4(3) v5(4) u6(5) v6(6) u7(7) v7(8) v8(9) u9(10) v9(11) u10(12) v10(13)

Displacement_complete= [0; 0; Displacement(1); 0; Displacement(2); 0; Displacement(3); 0; 0; Displacement(4); Displacement(5); Displacement(6); Displacement(7); Displacement(8); 0; Displacement(9); Displacement(10);Displacement(11); Displacement(12); Displacement(13)];

Reaction_Forces= K_complete*Displacement_complete;

x_deformed= [x(1); Displacement(1); Displacement(2); Displacement(3); x(5); Displacement(5); Displacement(7); x(8); Displacement(10); Displacement(12)];
y_deformed=[y(1); y(2); y(3); y(4); Displacement(4); Displacement(6); Displacement(8); Displacement(9); Displacement(11); Displacement(13)];

%calculating stresses and strains at 4 gauss points of element A aka element 3 in my case-->

disp_element_A= [Displacement(12); Displacement(13); Displacement(7); Displacement(8); Displacement(2); 0; Displacement(3); 0 ];% arranged anti-clockwise as per chp5 slide

strain_1= B31*disp_element_A;
strain_2= B32*disp_element_A;
strain_3= B33*disp_element_A;
strain_4= B34*disp_element_A;

stress_1= E*strain_1;
stress_2= E*strain_2;
stress_3= E*strain_3;
stress_4= E*strain_4;

Complete_stresses = [stress_1(1:2,:) stress_2(1:2,:) stress_3(1:2,:) stress_4(1:2,:)];% arranged anti-clockwise as per chp5 slide
Complete_strain = [strain_1(1:2,:) strain_2(1:2,:) strain_3(1:2,:) strain_4(1:2,:)];% arranged anti-clockwise as per chp5 slide

figure(1)
scatter(x,y, 'filled')
hold on
scatter(x_deformed, y_deformed, 'filled')
xlabel("x-coordinates of element nodes")
ylabel("y-coordinates of element nodes")
legend("Initial node condition", "Final node condition")
hold off

% figure(2)
% Reaction_Forces_X= [Forces(1), Forces(3), Forces(5),Forces(7),Forces(9),Forces(11),Forces(13),Forces(15),Forces(17),Forces(19)];
% Reaction_Forces_Y= [Forces(2), Forces(4), Forces(6),Forces(8),Forces(10),Forces(12),Forces(14),Forces(16),Forces(18),Forces(20)];
% quiver(Reaction_Forces_X, Reaction_Forces_Y)

